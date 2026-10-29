$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$hfExe = Join-Path $projectRoot '.venv\Scripts\hf.exe'
$models = (Get-Content 'results\environment.json' -Raw | ConvertFrom-Json).models

$files = @{
    llama = @('model.safetensors', 'config.json', 'generation_config.json',
              'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json')
    gemma = @('model.safetensors', 'config.json', 'generation_config.json',
              'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
              'tokenizer.model', 'added_tokens.json')
}
foreach ($modelName in @('llama', 'gemma')) {
    $entry = $models.$modelName
    $selected = $files[$modelName]
    Write-Output "Caching $modelName at $($entry.revision)"
    & $hfExe download $entry.id @selected --revision $entry.revision
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $modelName" }
}
Write-Output 'Both gated model caches are ready.'
