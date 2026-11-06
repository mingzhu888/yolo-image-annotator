@echo off
setlocal
cd /d "%~dp0"

set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 7\ISCC.exe"
if not exist "%ISCC%" (
  echo Inno Setup compiler not found:
  echo %ISCC%
  pause
  exit /b 1
)

if not exist "dist\云企智瞰标注工具\云企智瞰标注工具.exe" (
  echo Build the EXE first:
  echo build_exe.bat
  pause
  exit /b 1
)

if not exist "installer\redist\MicrosoftEdgeWebview2Setup.exe" goto :missing
if not exist "installer\redist\vc_redist.x64.exe" goto :missing

"%ISCC%" "installer\YoloAnnotator.iss"
if errorlevel 1 goto :failed

echo.
echo Installer created in installer\output
pause
exit /b 0

:missing
echo Runtime prerequisites are missing in installer\redist.
pause
exit /b 1

:failed
echo.
echo Installer build failed.
pause
exit /b 1
