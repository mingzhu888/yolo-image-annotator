# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules

hiddenimports = collect_submodules("ultralytics")
webview_datas, webview_binaries, webview_hiddenimports = collect_all("webview")
hiddenimports += webview_hiddenimports + ["clr", "pythonnet", "clr_loader"]

a = Analysis(
    ["desktop_app.py"],
    pathex=[],
    binaries=webview_binaries,
    datas=webview_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "matplotlib.tests",
        "numpy.tests",
        "torch.test",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="云企智瞰标注工具",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="assets/yunqi-zhikan-annotator.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="云企智瞰标注工具",
)
