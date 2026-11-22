@echo off
setlocal
REM ============================================================
REM  sync_from_4060.bat  -  incremental pull from the 4060 desktop
REM  Edit SRC_IP / SHARE_NAME to match the desktop SMB share.
REM  Only features, logs and outputs are synced. No deletion.
REM ============================================================
set SRC_IP=192.168.137.1
set SHARE_NAME=ColonMultiModal
set SRC=\\%SRC_IP%\%SHARE_NAME%
set DST=%~dp0
set LOG=%DST%sync_from_4060.log

echo [INFO] source = %SRC%
echo [INFO] dest   = %DST%
ping -n 1 -w 1000 %SRC_IP% >nul
if errorlevel 1 (
  echo [ERROR] Cannot reach %SRC_IP%. Check hotspot / LAN and the IP.
  pause
  exit /b 1
)

robocopy "%SRC%\features" "%DST%features" /E /R:1 /W:1 /NP /XD __pycache__ /LOG+:"%LOG%" /TEE
if errorlevel 8 goto :failed
robocopy "%SRC%\logs" "%DST%logs" /E /R:1 /W:1 /NP /XD __pycache__ /LOG+:"%LOG%" /TEE
if errorlevel 8 goto :failed
robocopy "%SRC%\outputs" "%DST%outputs" /E /R:1 /W:1 /NP /XD __pycache__ /LOG+:"%LOG%" /TEE
if errorlevel 8 goto :failed

echo [DONE] sync finished. Log: %LOG%
pause
exit /b 0

:failed
echo [ERROR] robocopy failed, see %LOG%
pause
exit /b 1
