$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc1 = (Resolve-Path "outputs\Task1_BestBuy_Report.docx").Path
    $pdf1 = [System.IO.Path]::ChangeExtension($doc1, ".pdf")
    $d1 = $word.Documents.Open($doc1)
    $d1.SaveAs([ref]$pdf1, [ref]17)
    $d1.Close()
    Write-Host "Task1 PDF created: $pdf1"

    $doc2 = (Resolve-Path "outputs\Task2_Stakeholder_Report.docx").Path
    $pdf2 = [System.IO.Path]::ChangeExtension($doc2, ".pdf")
    $d2 = $word.Documents.Open($doc2)
    $d2.SaveAs([ref]$pdf2, [ref]17)
    $d2.Close()
    Write-Host "Task2 PDF created: $pdf2"
} catch {
    Write-Error $_
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
