@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Raw $env:KINGOGPT_SSH_PW_FILE"
