# -*- coding: utf-8 -*-
"""从品牌 PNG 生成 Windows 多尺寸 ICO，不覆盖原始 logo。"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
PNG = ASSETS / "yunqi-zhikan-annotator.png"
ICO = ASSETS / "yunqi-zhikan-annotator.ico"


def main():
    if not PNG.exists():
        raise FileNotFoundError(f"品牌 PNG 不存在: {PNG}")

    image = Image.open(PNG).convert("RGBA")
    image.save(
        ICO,
        sizes=[
            (256, 256),
            (128, 128),
            (64, 64),
            (48, 48),
            (32, 32),
            (16, 16),
        ],
    )
    print(ICO)


if __name__ == "__main__":
    main()
