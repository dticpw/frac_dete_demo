$ErrorActionPreference = "Stop"

$python = "D:/python/anaconda/envs/fracmed/python.exe"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $scriptDir
& $python app.py
