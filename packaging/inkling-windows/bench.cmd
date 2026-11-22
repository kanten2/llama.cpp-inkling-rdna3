@echo off
setlocal DisableDelayedExpansion
set "INKLING_MODEL=%~f1"
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\launch.ps1" -Mode Bench
set "result=%ERRORLEVEL%"
if not defined INKLING_NO_PAUSE pause >nul
exit /b %result%
