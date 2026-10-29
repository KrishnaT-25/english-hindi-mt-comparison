$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$gemmaPartial = Join-Path $projectRoot 'results\predictions\gemma_test_greedy.jsonl'

Add-Type -Namespace Win32 -Name PowerRequest500 -MemberDefinition @'
    [System.Runtime.InteropServices.DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
'@
[void][Win32.PowerRequest500]::SetThreadExecutionState(
    [uint32]::Parse('80000001', [Globalization.NumberStyles]::HexNumber))

function Invoke-ExperimentCommand([string[]] $arguments) {
    Write-Output "Running: python $($arguments -join ' ')"
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

$gemmaSubset = Join-Path $projectRoot 'results\predictions\gemma_test_500_greedy.jsonl'
if (Test-Path -LiteralPath $gemmaPartial) {
    $savedCount = (Get-Content -LiteralPath $gemmaPartial | Measure-Object -Line).Lines
    if ($savedCount -lt 500) {
        Write-Output 'Resuming Gemma through the first 500 frozen test IDs...'
        Invoke-ExperimentCommand -arguments @('-m', 'src.generate', 'gemma', '--limit', '500')
    } elseif ($savedCount -ne 500) {
        throw "Gemma parent run has $savedCount saved rows; expected exactly 500."
    }
    Invoke-ExperimentCommand -arguments @('-m', 'src.subset_test', '--count', '500')
    # Keep the interrupted 1,000-example parent file out of published results.
    Move-Item -LiteralPath $gemmaPartial -Destination (Join-Path $projectRoot '.tmp\gemma_test_partial_500.jsonl') -Force
    $gemmaParentMetadata = Join-Path $projectRoot 'results\metrics\gemma_test_greedy_run.json'
    if (Test-Path -LiteralPath $gemmaParentMetadata) {
        Move-Item -LiteralPath $gemmaParentMetadata -Destination (Join-Path $projectRoot '.tmp\gemma_test_partial_run.json') -Force
    }
} elseif (-not (Test-Path -LiteralPath $gemmaSubset)) {
    throw 'Gemma has neither a resumable parent run nor a complete 500-example subset.'
}
foreach ($modelName in @('transformer', 'mt5', 'llama', 'gemma')) {
    $predictionFile = "results/predictions/${modelName}_test_500_greedy.jsonl"
    Invoke-ExperimentCommand -arguments @('-m', 'src.evaluate', $predictionFile, '--subset-size', '500')
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
Invoke-ExperimentCommand -arguments @('-m', 'src.analyze', '--test-count', '500')
Invoke-ExperimentCommand -arguments @('-m', 'src.report', '--test-count', '500')
Write-Output 'Shared 500-example comparison and all temperature runs completed.'
[void][Win32.PowerRequest500]::SetThreadExecutionState(
    [uint32]::Parse('80000000', [Globalization.NumberStyles]::HexNumber))
