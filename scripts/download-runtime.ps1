$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $root '.venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'download-runtime.py')
exit $LASTEXITCODE
