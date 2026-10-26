"""Command-line entry point for the HanuAi ML assessment solution.

Examples
--------
    .venv\\Scripts\\python.exe run_pipeline.py
    .venv\\Scripts\\python.exe run_pipeline.py --live
    .venv\\Scripts\\python.exe run_pipeline.py --skip-notebooks
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hanuai.pipeline import run_all


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--live",
        action="store_true",
        help="Re-run the live Best Buy scraping demonstration instead of using the cached snapshot.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of K-Means clusters for Task 2 (default: 5).",
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Build datasets and reports without regenerating/executing the notebooks.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip Word report generation.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    result = run_all(
        live=args.live,
        n_clusters=args.clusters,
        build_notebooks=not args.skip_notebooks,
        make_reports=not args.skip_reports,
    )

    task1 = result["task1"]
    task2 = result["task2"]

    print("\n=== Task 1 ===")
    print(f"Graded reviews : {len(task1['graded'])}")
    print(f"Demo reviews   : {len(task1['demo'])}")
    print(f"Dataset rows   : {task1['rows']}")
    print(f"Dataset        : {task1['dataset_path']}")

    print("\n=== Task 2 ===")
    print(f"Records        : {task2['rows']}")
    print(f"Clusters       : {task2['clustering'].n_clusters} ({task2['clustering'].backend})")
    print(f"Dataset        : {task2['dataset_path']}")

    if result["reports"]:
        print("\n=== Reports ===")
        for name, path in result["reports"].items():
            print(f"{name}: {path} ({Path(path).stat().st_size / 1024:.0f} KB)")

    if result["notebooks"]:
        print("\n=== Notebooks ===")
        for name, path in result["notebooks"].items():
            print(f"{name}: {path} ({Path(path).stat().st_size / 1024:.0f} KB)")

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
