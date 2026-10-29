$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$hfExe = Join-Path $projectRoot '.venv\Scripts\hf.exe'
$mt5Metric = Join-Path $projectRoot 'results\metrics\mt5_test_greedy.json'

# Prevent idle system sleep while this worker owns the long GPU runs.
Add-Type -Namespace Win32 -Name PowerRequest -MemberDefinition @'
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
'@
[void][Win32.PowerRequest]::SetThreadExecutionState(
    [uint32]::Parse('80000001', [Globalization.NumberStyles]::HexNumber))

function Invoke-ExperimentCommand([string[]] $arguments) {
    Write-Output "Running: python $($arguments -join ' ')"
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

Write-Output 'Waiting for mT5 test metrics and Hugging Face gated-model access...'
for ($attempt = 0; $attempt -lt 288; $attempt++) {
    if (Test-Path -LiteralPath $mt5Metric) {
        $who = & $hfExe auth whoami 2>$null
        if ($who -and ($who -notmatch 'Not logged in')) {
            Invoke-ExperimentCommand -arguments @('-m', 'src.check_access')
            $access = Get-Content 'results\model_access.json' -Raw | ConvertFrom-Json
            if ($access.llama.accessible -and $access.gemma.accessible) { break }
            Write-Output 'Authenticated, but Llama or Gemma terms/access are still pending.'
        }
    }
    Start-Sleep -Seconds 300
}
if (-not (Test-Path -LiteralPath $mt5Metric)) { throw 'mT5 test metrics were not ready within 24 hours.' }
if (-not ($access.llama.accessible -and $access.gemma.accessible)) {
    throw 'Llama and Gemma access was not available within 24 hours.'
}

foreach ($modelName in @('llama', 'gemma')) {
    $predictionFile = "results/predictions/${modelName}_test_greedy.jsonl"
    if (-not (Test-Path -LiteralPath $predictionFile)) {
        Invoke-ExperimentCommand -arguments @('-m', 'src.generate', $modelName, '--limit', '1')
    }
    Invoke-ExperimentCommand -arguments @('-m', 'src.generate', $modelName)
    Invoke-ExperimentCommand -arguments @('-m', 'src.evaluate', $predictionFile)
    Invoke-ExperimentCommand -arguments @('-m', 'src.benchmark', $predictionFile)
}

foreach ($modelName in @('transformer', 'mt5', 'llama', 'gemma')) {
    foreach ($temperature in @('greedy', '0.3', '0.7')) {
        $arguments = @('-m', 'src.generate', $modelName, '--split', 'validation', '--limit', '100')
        if ($temperature -ne 'greedy') { $arguments += @('--temperature', $temperature) }
        Invoke-ExperimentCommand -arguments $arguments
        $suffix = if ($temperature -eq 'greedy') { 'greedy' } else { "t$temperature" }
        $predictionFile = "results/predictions/${modelName}_validation_${suffix}.jsonl"
        Invoke-ExperimentCommand -arguments @('-m', 'src.evaluate', $predictionFile, '--split', 'validation')
    }
}
Invoke-ExperimentCommand -arguments @('-m', 'src.analyze')
Invoke-ExperimentCommand -arguments @('-m', 'src.report')
Write-Output 'All four models and the validation temperature study completed.'
[void][Win32.PowerRequest]::SetThreadExecutionState(
    [uint32]::Parse('80000000', [Globalization.NumberStyles]::HexNumber))
