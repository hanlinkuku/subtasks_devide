$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
Write-Host 'Annotation workbench: http://127.0.0.1:8765'
python workbench.py
