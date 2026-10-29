param([int]$TestCount = 500)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$reportFile = Join-Path $projectRoot 'results\REPORT.md'

Write-Output 'Waiting for all four model runs and the report...'
for ($attempt = 0; $attempt -lt 288; $attempt++) {
    if (Test-Path -LiteralPath $reportFile) { break }
    Start-Sleep -Seconds 300
}
if (-not (Test-Path -LiteralPath $reportFile)) {
    throw 'The full report was not ready within 24 hours.'
}
Start-Sleep -Seconds 60

foreach ($modelName in @('transformer', 'mt5', 'llama', 'gemma')) {
    $predictionFile = "results/predictions/${modelName}_test_${TestCount}_greedy.jsonl"
    & $pythonExe -m src.evaluate $predictionFile --subset-size $TestCount
    if ($LASTEXITCODE -ne 0) { throw "Test score verification failed: $modelName" }
    foreach ($suffix in @('greedy', 't0.3', 't0.7')) {
        $predictionFile = "results/predictions/${modelName}_validation_${suffix}.jsonl"
        & $pythonExe -m src.evaluate $predictionFile --split validation
        if ($LASTEXITCODE -ne 0) { throw "Validation score verification failed: $modelName $suffix" }
    }
}
& $pythonExe -m src.report --test-count $TestCount
if ($LASTEXITCODE -ne 0) { throw 'Final report regeneration failed.' }

git add README.md results "data/manifests/test_${TestCount}_ids.json"
if ($LASTEXITCODE -ne 0) { throw 'Could not stage final results.' }
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m 'Publish measured four-model translation comparison'
    if ($LASTEXITCODE -ne 0) { throw 'Could not commit final results.' }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Could not push final results.' }
}
Write-Output 'Verified scores and published the final result files.'
