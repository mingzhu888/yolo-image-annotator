@echo off
chcp 65001 >nul
title YOLO 标注工具
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" -c "import flask, PIL" >nul 2>nul
if errorlevel 1 (
    echo 缺少依赖，请先双击 install_deps.bat 安装依赖。
    pause
    exit /b 1
)

echo ============================================
echo   正在启动 YOLO 标注工具...
echo   浏览器会自动打开，请稍候
echo   关闭本窗口即可退出工具
echo ============================================
"%PY%" annotate_tool.py
pause
