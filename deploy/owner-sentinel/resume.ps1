$pause = Join-Path $env:LOCALAPPDATA 'nakshatra\pause'; if (Test-Path -LiteralPath $pause) { Remove-Item -LiteralPath $pause -Force }
