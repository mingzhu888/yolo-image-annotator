# -*- coding: utf-8 -*-
"""只读:检测图片数据集中的重复/近重复图片。不删除任何文件。"""
import argparse
import hashlib
import os
import re
from collections import defaultdict

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(im, size=8):
    im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(im.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            left = px[r * (size + 1) + c]
            right = px[r * (size + 1) + c + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def main():
    ap = argparse.ArgumentParser(
        description="检测重复/近重复图片(只读,不删除任何文件)")
    ap.add_argument("--img-dir", required=True, help="图片目录")
    ap.add_argument("--report", default="",
                    help="报告输出路径(默认 图片目录/重复图片报告.txt)")
    ap.add_argument("--limit", type=int, default=60,
                    help="同源组最多列出多少组(默认 60)")
    args = ap.parse_args()

    D = os.path.abspath(args.img_dir)
    REPORT = os.path.abspath(args.report or os.path.join(D, "重复图片报告.txt"))
    LIMIT = args.limit

    files = sorted(f for f in os.listdir(D)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    print(f"总图片数: {len(files)}")

    md5_map = defaultdict(list)      # 字节级完全相同
    dhash_map = defaultdict(list)    # 像素级完全相同(解码后)
    prefix_map = defaultdict(list)   # Roboflow 同源原图
    bad = []

    for i, name in enumerate(files):
        if i % 1000 == 0:
            print(f"  处理 {i}/{len(files)} ...")
        path = os.path.join(D, name)
        # 原图前缀 (去掉 _jpg.rf.xxxx.jpg)
        m = re.match(r"(.+?)_jpg\.rf\.", name)
        prefix = m.group(1) if m else name
        prefix_map[prefix].append(name)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
            md5_map[hashlib.md5(data).hexdigest()].append(name)
            with Image.open(path) as im:
                dhash_map[dhash(im)].append(name)
        except Exception as e:
            bad.append((name, str(e)))

    def groups_over1(mp):
        return {k: v for k, v in mp.items() if len(v) > 1}

    exact = groups_over1(md5_map)
    visual = groups_over1(dhash_map)
    samesrc = groups_over1(prefix_map)

    exact_extra = sum(len(v) - 1 for v in exact.values())
    visual_extra = sum(len(v) - 1 for v in visual.values())
    src_extra = sum(len(v) - 1 for v in samesrc.values())

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 60)
    out(f"数据集: {D}")
    out(f"总图片数: {len(files)}")
    out("-" * 60)
    out(f"[1] 字节完全相同(真·重复): {len(exact)} 组, 多出 {exact_extra} 张冗余")
    out(f"[2] 像素解码后相同(重新编码的重复): {len(visual)} 组, 多出 {visual_extra} 张")
    out(f"[3] 同一原图(Roboflow 同源/增广): {len(samesrc)} 组, 多出 {src_extra} 张")
    if bad:
        out(f"[!] 无法读取的文件: {len(bad)} 个")
    out("=" * 60)

    out("\n########## [1] 字节完全相同(强烈建议只留1张) ##########")
    for h, names in sorted(exact.items(), key=lambda x: -len(x[1])):
        out(f"\n[{len(names)} 张相同] md5={h[:12]}")
        for n in names:
            out(f"    {n}")

    out("\n\n########## [2] 像素相同但字节不同(重新编码/改元数据) ##########")
    exact_names = set(n for v in exact.values() for n in v)
    cnt2 = 0
    for h, names in sorted(visual.items(), key=lambda x: -len(x[1])):
        if set(names) <= exact_names:
            continue
        cnt2 += 1
        out(f"\n[{len(names)} 张画面相同]")
        for n in names:
            out(f"    {n}")
    if cnt2 == 0:
        out("  (无:像素级重复都已被[1]覆盖)")

    out("\n\n########## [3] 同一原图前缀(可能是增广或重复导出) ##########")
    for p, names in sorted(samesrc.items(), key=lambda x: -len(x[1]))[:LIMIT]:
        out(f"\n[原图 {p} -> {len(names)} 张]")
        for n in names:
            out(f"    {n}")
    if len(samesrc) > LIMIT:
        out(f"\n  ... 还有 {len(samesrc) - LIMIT} 组同源图未全部列出(见完整报告)")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    out(f"\n完整报告已写入: {REPORT}")


if __name__ == "__main__":
    main()
