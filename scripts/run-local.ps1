param(
    [ValidateSet('run','generate','verify','export','status','doctor')]
    [string]$Command = 'run',
    [string]$Config = 'configs/local-4bit.json'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Missing .venv. Follow the local installation section in README.md.'
}
$env:HF_HOME = Join-Path $projectRoot '.hf_cache'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -B -m vgec $Command --config $Config
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
