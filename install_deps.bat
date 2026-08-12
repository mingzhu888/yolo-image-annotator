@echo off
chcp 65001 >nul
title YOLO 标注工具 - 安装依赖
cd /d "%~dp0"

echo ============================================
echo   正在安装依赖...
echo   包含 ultralytics / PyTorch，可能需要较长时间
echo   网络不好时建议使用代理后重试
echo ============================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境 .venv ...
    %PY% -m venv .venv
    if errorlevel 1 goto :err
)

echo [2/3] 升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :err

echo [3/3] 安装依赖 ...
".venv\Scripts\python.exe" -m pip install -r requirements-full.txt
if errorlevel 1 goto :err

echo.
echo ============================================
echo   安装完成！双击 start.bat 即可启动工具
echo ============================================
pause
exit /b 0

:err
echo.
echo 安装失败。请检查：
echo  1. 已安装 Python 3.9 - 3.12（64 位）
echo  2. 网络可访问 pypi.org
echo  3. 磁盘空间充足（PyTorch 需要数 GB）
pause
exit /b 1
