$pause = Join-Path $env:LOCALAPPDATA 'nakshatra\pause'; New-Item -ItemType Directory -Force -Path (Split-Path $pause) | Out-Null; New-Item -ItemType File -Force -Path $pause | Out-Null
