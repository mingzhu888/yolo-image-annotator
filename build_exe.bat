@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=F:\miniconda\envs\yolo_4060\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Python environment not found:
  echo %PYTHON_EXE%
  pause
  exit /b 1
)

echo Installing build dependency...
"%PYTHON_EXE%" -m pip install pyinstaller
if errorlevel 1 goto :failed

echo Building YOLO annotator...
"%PYTHON_EXE%" -m PyInstaller --noconfirm --clean YoloAnnotator.spec
if errorlevel 1 goto :failed

echo.
echo Build complete:
echo dist\YOLO标注工具\YOLO标注工具.exe
pause
exit /b 0

:failed
echo.
echo Build failed.
pause
exit /b 1
