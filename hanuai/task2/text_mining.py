"""Regex/keyword entity tagging and issue categorisation (Task 2).

The miner works on the three VERBATIM columns and requires no external NLP
models.  It normalises common Spanish terms to English before dictionary
matching, extracts components, symptoms and error codes, preserves the
context of redacted numeric codes, and assigns one or more issue categories.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

import pandas as pd

from ..config import ISSUE_CATEGORIES, VERBATIM_COLUMNS

# Spanish -> English normalisation for dictionary matching.  Longer phrases
# are applied first so "no funciona" wins over "funciona".
SPANISH_TERMS: dict[str, str] = {
    "pantalla se pone negra": "black screen",
    "no funciona": "inoperative",
    "no enciende": "no power",
    "se apaga": "shuts off",
    "se reinicia": "reboots",
    "pérdida de comunicación": "loss of communication",
    "perdida de comunicacion": "loss of communication",
    "cortocircuito": "short circuit",
    "actualización": "update",
    "actualizacion": "update",
    "reprogramar": "reprogram",
    "reemplazar": "replace",
    "reemplazo": "replace",
    "reparar": "repair",
    "revisar": "inspect",
    "pantallas": "display",
    "pantalla": "display",
    "módulos": "module",
    "modulos": "module",
    "módulo": "module",
    "modulo": "module",
    "batería": "battery",
    "bateria": "battery",
    "cámaras": "camera",
    "camara": "camera",
    "cámara": "camera",
    "cables": "wiring",
    "cable": "wiring",
    "conector": "connector",
    "arnés": "harness",
    "arnes": "harness",
    "tablero": "cluster",
    "sonido": "sound",
    "ruido": "noise",
    "fallas": "failure",
    "falla": "failure",
    "negro": "black",
    "negra": "black",
    "obscura": "dark",
    "reversa": "reverse",
    "cliente": "customer",
    "problema": "problem",
    "transmisión": "transmission",
    "transmision": "transmission",
    "motor": "engine",
    "funcionamiento": "operation",
    "comunicación": "communication",
    "comunicacion": "communication",
    "señal": "signal",
    "senal": "signal",
    "sistema": "system",
    "funciona": "works",
}

COMPONENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Radio": ("radio", "head unit", "receiver", "audio unit", "infotainment"),
    "Display": ("display", "screen", "touchscreen", "touch screen", "monitor"),
    "Amplifier": ("amplifier", "amp"),
    "Speaker": ("speaker", "speakers"),
    "Radio Module": ("radio module", "telematics communication", "tcip", "telematics module"),
    "OnStar": ("onstar",),
    "Navigation": ("navigation", "nav unit", "gps", "satellite radio"),
    "Bluetooth": ("bluetooth",),
    "USB / SD": ("usb", "sd card", "media card"),
    "Camera": ("camera", "rear vision"),
    "Cluster": ("cluster", "instrument panel", "gauge"),
    "Wiring": ("wiring", "harness", "cable", "connector", "terminal", "pins"),
    "Antenna": ("antenna",),
    "Battery": ("battery",),
    "HVAC": ("hvac", "climate control", "blower"),
    "Engine": ("engine",),
    "Transmission": ("transmission",),
}

SYMPTOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Black / Blank Screen": (
        "black screen",
        "blank screen",
        "screen is blank",
        "screen was blank",
        "screen goes black",
        "no display",
        "display is black",
    ),
    "Inoperative": (
        "inoperative",
        "inop",
        "does not work",
        "doesn't work",
        "not working",
        "no function",
        "no power",
    ),
    "No Sound": ("no sound", "no audio", "audio inop", "muted"),
    "Intermittent": ("intermittent", "intermittently", "intermittant", "sporadic", "sometimes"),
    "Reset / Reboot": ("reset", "reboot", "restarts", "shuts off", "turns off", "cycles", "power cycle"),
    "Frozen / Unresponsive": (
        "freeze",
        "frozen",
        "lock up",
        "locked up",
        "unresponsive",
        "irresponsive",
        "does not respond",
    ),
    "Noise / Distortion": ("static", "distortion", "buzzing", "crackle", "humming", "abnormal noise"),
    "Communication Loss": (
        "lost communication",
        "loss of communication",
        "no communication",
        "communication loss",
    ),
    "Error Message": ("error message", "error appears", "error code", "prompt"),
    "Battery Drain": ("battery drain", "parasitic draw", "dead battery", "battery dies"),
    "Programming Failure": (
        "programming error",
        "programming failed",
        "unable to program",
        "sps error",
        "no new updated software",
    ),
    "Short Circuit": ("short circuit", "shorted", "short to"),
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Software Glitch": (
        "software",
        "reprogram",
        "reprogrammed",
        "programming",
        "programmed",
        "update",
        "updated",
        "calibration",
        "reset",
        "reboot",
        "freeze",
        "frozen",
        "glitch",
        "configuration",
        "ota",
        "sps",
        "registers incorrectly",
        "usb programming",
        "module setup",
    ),
    "Electrical Issue": (
        "wiring",
        "short",
        "shorted",
        "short circuit",
        "circuit",
        "battery",
        "voltage",
        "ground",
        "connector",
        "fuse",
        "harness",
        "terminal",
        "pins",
        "ohm",
        "resistance",
        "parasitic draw",
        "no power",
        "power supply",
        "antenna",
    ),
    "Assembly Quality": (
        "loose",
        "rattle",
        "squeak",
        "fit",
        "finish",
        "panel",
        "trim",
        "misalign",
        "gap",
        "clip",
        "bolt",
        "screw",
        "assembly",
        "build quality",
        "poor fitment",
        "vibration",
        "worn",
        "stripped",
        "poor contact",
    ),
    "Component Failure": (
        "failed",
        "failure",
        "faulty",
        "broken",
        "inoperative",
        "inop",
        "no sound",
        "black screen",
        "blank",
        "dead",
        "defective",
        "malfunction",
        "internal fault",
        "not working",
        "does not work",
        "no display",
    ),
}

CAUSAL_CODE_CATEGORY: dict[str, str] = {
    "shorted": "Electrical Issue",
    "no/incorrect communication": "Electrical Issue",
    "poor contact": "Electrical Issue",
    "open circuit": "Electrical Issue",
    "registers incorrectly": "Software Glitch",
    "reprogrammed": "Software Glitch",
    "tech bulletin": "Software Glitch",
    "campaign": "Software Glitch",
    "broken": "Component Failure",
    "worn/stripped": "Assembly Quality",
    "no trouble found": "Software Glitch",
}

_DTC_RE = re.compile(r"\b[PBCU]\d{3,4}\b", re.IGNORECASE)
_NAMED_CODE_RE = re.compile(r"\b(?:sps|dtc|ota)\b", re.IGNORECASE)
_REDACTED_RE = re.compile(
    r"\b(?:op\s*code|warranty\s+code|codes?|dtcs?|sps|error)\b"
    r"(?:[ \t\u00a0\u2009]{2,}|[=:][ \t\u00a0\u2009]{2,})",
    re.IGNORECASE,
)
# Terms that are distinctive enough not to appear in ordinary English text.
_SPANISH_HINTS = (
    "pantalla",
    "pantallas",
    "falla",
    "fallas",
    "funciona",
    "no funciona",
    "bateria",
    "batería",
    "modulo",
    "módulo",
    "modulos",
    "módulos",
    "sonido",
    "ruido",
    "negro",
    "negra",
    "obscura",
    "cliente",
    "problema",
    "revisar",
    "reparar",
    "reemplazar",
    "comunicación",
    "comunicacion",
    "señal",
    "senal",
    "tablero",
    "arnés",
    "arnes",
    "cámara",
    "camara",
    "transmisión",
    "transmision",
    "cortocircuito",
    "actualización",
    "actualizacion",
    "reprogramar",
    "palanca",
    "velocidades",
)



def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(keyword) + r"(?:s|es|ed|ing)?\b", re.IGNORECASE)


_COMPONENT_PATTERNS = {
    name: [_keyword_pattern(keyword) for keyword in keywords]
    for name, keywords in COMPONENT_KEYWORDS.items()
}
_SYMPTOM_PATTERNS = {
    name: [_keyword_pattern(keyword) for keyword in keywords]
    for name, keywords in SYMPTOM_KEYWORDS.items()
}
_CATEGORY_PATTERNS = {
    name: [_keyword_pattern(keyword) for keyword in keywords]
    for name, keywords in CATEGORY_KEYWORDS.items()
}


def normalise_for_matching(text: str) -> str:
    """Normalise unicode, lower-case and translate Spanish terms to English."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u2009", " ").replace("\u00a0", " ").replace("\u2019", "'")
    value = re.sub(r"\s+", " ", value).strip().lower()
    for source in sorted(SPANISH_TERMS, key=len, reverse=True):
        target = SPANISH_TERMS[source]
        value = re.sub(r"\b" + re.escape(source) + r"\b", target, value)
    return value


def _matches(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def extract_components(text: str) -> list[str]:
    return [name for name, patterns in _COMPONENT_PATTERNS.items() if _matches(text, patterns)]


def extract_symptoms(text: str) -> list[str]:
    return [name for name, patterns in _SYMPTOM_PATTERNS.items() if _matches(text, patterns)]


def extract_error_codes(raw_text: str, normalised_text: str) -> list[str]:
    codes = {match.upper() for match in _DTC_RE.findall(raw_text)}
    codes.update(match.upper() for match in _NAMED_CODE_RE.findall(normalised_text))
    return sorted(codes)


def find_redacted_contexts(raw_text: str) -> list[str]:
    """Return short contexts around numeric codes redacted as whitespace."""
    contexts = []
    for match in _REDACTED_RE.finditer(raw_text):
        start = max(0, match.start() - 35)
        end = min(len(raw_text), match.end() + 20)
        snippet = raw_text[start:end]
        # Replace the whitespace run with an explicit marker before collapsing
        # the remaining spacing, so the redaction stays visible.
        snippet = _REDACTED_RE.sub(lambda m: m.group(0).rstrip() + " [REDACTED] ", snippet)
        snippet = re.sub(r"[ \t\u00a0\u2009]{2,}", " ", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        if snippet:
            contexts.append(snippet)
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = []
    for snippet in contexts:
        if snippet not in seen:
            seen.add(snippet)
            unique.append(snippet)
    return unique[:5]


def _category_scores(normalised_text: str) -> dict[str, int]:
    """Count distinct keyword hits per issue category."""
    return {
        name: sum(len(pattern.findall(normalised_text)) for pattern in patterns)
        for name, patterns in _CATEGORY_PATTERNS.items()
    }


def assign_issue_categories(normalised_text: str, causal_code: str = "") -> list[str]:
    """Assign one or more issue categories, falling back to causal-code hints.

    The category with the most keyword evidence becomes the primary category;
    any other category with at least two hits is also attached.
    """
    scores = _category_scores(normalised_text)
    causal = str(causal_code or "").lower()
    for fragment, category in CAUSAL_CODE_CATEGORY.items():
        if fragment in causal:
            scores[category] = scores.get(category, 0) + 2

    selected = [name for name in ISSUE_CATEGORIES if scores.get(name, 0) >= 2]
    if not selected:
        selected = [name for name in ISSUE_CATEGORIES if scores.get(name, 0) >= 1]
    if not selected:
        selected = ["Component Failure"]

    # Primary first, then the remaining categories in canonical order.
    primary = max(selected, key=lambda name: (scores.get(name, 0), -ISSUE_CATEGORIES.index(name)))
    ordered = [primary] + [name for name in ISSUE_CATEGORIES if name in selected and name != primary]
    return ordered


def _as_text(value: Any) -> str:
    """Coerce a cell to text without turning missing values into 'nan'."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value)


def mine_record(row: pd.Series) -> dict[str, Any]:
    """Extract entities and categories for one dataset record."""
    raw_combined = " ".join(
        _as_text(row.get(column)) for column in VERBATIM_COLUMNS
    ).strip()
    normalised = normalise_for_matching(raw_combined)
    components = extract_components(normalised)
    symptoms = extract_symptoms(normalised)
    codes = extract_error_codes(raw_combined, normalised)
    redacted = find_redacted_contexts(raw_combined)
    categories = assign_issue_categories(normalised, str(row.get("CAUSAL_CD_DESC") or ""))
    spanish = any(re.search(r"\b" + re.escape(term) + r"\b", raw_combined.lower()) for term in _SPANISH_HINTS)
    return {
        "NormalisedVerbatim": normalised,
        "Components": components,
        "Symptoms": symptoms,
        "ErrorCodes": codes,
        "RedactedCodeFlag": bool(redacted),
        "RedactedCodeContexts": redacted,
        "SpanishFlag": bool(spanish),
        "IssueCategories": categories,
        "PrimaryIssueCategory": categories[0],
    }


def mine_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Add entity-tag and issue-category columns to a complaint frame."""
    mined = frame.copy()
    records = [mine_record(row) for _, row in mined.iterrows()]
    extracted = pd.DataFrame.from_records(records)
    for column in extracted.columns:
        mined[column] = extracted[column].values

    mined["ComponentsText"] = mined["Components"].map(lambda values: "|".join(values))
    mined["SymptomsText"] = mined["Symptoms"].map(lambda values: "|".join(values))
    mined["ErrorCodesText"] = mined["ErrorCodes"].map(lambda values: "|".join(values))
    mined["RedactedCodeContextsText"] = mined["RedactedCodeContexts"].map(
        lambda values: " | ".join(values)
    )
    mined["IssueCategory"] = mined["IssueCategories"].map(lambda values: "|".join(values))
    mined["EntityTags"] = mined.apply(
        lambda row: "; ".join(
            part
            for part in (
                f"Components: {row['ComponentsText']}" if row["ComponentsText"] else "",
                f"Symptoms: {row['SymptomsText']}" if row["SymptomsText"] else "",
                f"Codes: {row['ErrorCodesText']}" if row["ErrorCodesText"] else "",
            )
            if part
        ),
        axis=1,
    )
    return mined


def mining_summary(mined: pd.DataFrame) -> dict[str, Any]:
    """Aggregate entity and category frequencies for reporting."""
    def counter(column: str) -> Counter:
        result: Counter = Counter()
        for values in mined[column]:
            result.update(values)
        return result

    category_counts = counter("IssueCategories")
    primary_counts = mined["PrimaryIssueCategory"].value_counts()
    components = counter("Components")
    symptoms = counter("Symptoms")
    codes = counter("ErrorCodes")

    def top(counter_obj: Counter, limit: int = 12) -> list[dict[str, Any]]:
        return [{"name": name, "count": count} for name, count in counter_obj.most_common(limit)]

    return {
        "category_counts": {
            name: int(category_counts.get(name, 0)) for name in ISSUE_CATEGORIES
        },
        "primary_counts": {
            name: int(primary_counts.get(name, 0)) for name in ISSUE_CATEGORIES
        },
        "multi_category_records": int((mined["IssueCategories"].map(len) > 1).sum()),
        "components": top(components),
        "symptoms": top(symptoms),
        "error_codes": top(codes),
        "redacted_records": int(mined["RedactedCodeFlag"].sum()),
        "spanish_records": int(mined["SpanishFlag"].sum()),
    }
