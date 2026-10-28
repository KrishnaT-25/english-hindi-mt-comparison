$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$trainingResult = Join-Path $projectRoot 'results\training_mt5.json'

Write-Output 'Waiting for mT5 training result...'
for ($attempt = 0; $attempt -lt 720; $attempt++) {
    if (Test-Path -LiteralPath $trainingResult) { break }
    Start-Sleep -Seconds 60
}
if (-not (Test-Path -LiteralPath $trainingResult)) {
    throw 'mT5 training did not finish within 12 hours. Inspect .tmp/mt5_train.err.log.'
}

# Let the trainer release GPU memory after writing its result file.
Start-Sleep -Seconds 120
foreach ($command in @(
    @('-m', 'src.generate', 'mt5'),
    @('-m', 'src.evaluate', 'results/predictions/mt5_test_greedy.jsonl'),
    @('-m', 'src.benchmark', 'results/predictions/mt5_test_greedy.jsonl')
)) {
    Write-Output "Running: python $($command -join ' ')"
    & $pythonExe @command
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}
Write-Output 'mT5 test generation, scoring, and benchmark completed.'
