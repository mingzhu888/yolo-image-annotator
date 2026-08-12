# -*- coding: utf-8 -*-
"""安全去重:把字节完全相同的重复图片+对应标注移动到隔离文件夹(不删除)。
默认预演模式,加 --apply 才真正移动。"""
import argparse
import csv
import hashlib
import os
import shutil
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(
        description="按 MD5 找出字节完全相同的重复图片并隔离(默认仅预演)")
    ap.add_argument("--img-dir", required=True, help="图片目录")
    ap.add_argument("--label-dir", default="", help="标注目录(默认与图片目录相同)")
    ap.add_argument("--img-out", default="",
                    help="重复图片隔离目录(默认 图片目录/重复图片隔离)")
    ap.add_argument("--label-out", default="",
                    help="重复标注隔离目录(默认 标注目录/标注_重复隔离)")
    ap.add_argument("--manifest", default="",
                    help="清单 CSV 路径(默认 图片目录/去重清单.csv)")
    ap.add_argument("--apply", action="store_true",
                    help="真正移动;不加则只预演")
    args = ap.parse_args()

    IMG = os.path.abspath(args.img_dir)
    LBL = os.path.abspath(args.label_dir or args.img_dir)
    IMG_Q = os.path.abspath(args.img_out or os.path.join(IMG, "重复图片隔离"))
    LBL_Q = os.path.abspath(args.label_out or os.path.join(LBL, "标注_重复隔离"))
    MANIFEST = os.path.abspath(args.manifest or os.path.join(IMG, "去重清单.csv"))
    APPLY = args.apply

    imgs = sorted(f for f in os.listdir(IMG)
                  if os.path.splitext(f)[1].lower() in IMG_EXTS)
    label_stems = set(os.path.splitext(f)[0] for f in os.listdir(LBL)
                      if f.lower().endswith(".txt"))

    def stem(name):
        return os.path.splitext(name)[0]

    def has_label(name):
        return stem(name) in label_stems

    # 1) 按 MD5 分组
    md5_map = defaultdict(list)
    for name in imgs:
        with open(os.path.join(IMG, name), "rb") as fh:
            md5_map[hashlib.md5(fh.read()).hexdigest()].append(name)

    dup_groups = {h: v for h, v in md5_map.items() if len(v) > 1}

    def keep_priority(name):
        # 数字越小越优先保留:已标注 > 无 b_ 前缀 > 名字短
        return (0 if has_label(name) else 1,
                1 if name.startswith("b_") else 0,
                len(name))

    to_move = []          # (img_name, reason_keep)
    kept_log = []
    for h, group in dup_groups.items():
        ordered = sorted(group, key=keep_priority)
        keep = ordered[0]
        kept_log.append((keep, len(group)))
        for name in ordered[1:]:
            to_move.append((name, keep))

    move_imgs = [m[0] for m in to_move]
    move_lbls = [m[0] for m in to_move if has_label(m[0])]

    kept_labeled = sum(1 for k, _ in kept_log if has_label(k))
    print("=" * 58)
    print(f"图片总数: {len(imgs)}")
    print(f"字节重复组数: {len(dup_groups)}")
    print(f"将移走的冗余图片: {len(move_imgs)} 张")
    print(f"  其中带标注(标注也一起移走): {len(move_lbls)} 个")
    print(f"去重后保留图片: {len(imgs) - len(move_imgs)} 张")
    print(f"  保留图中已标注: {kept_labeled} 张")
    b_moved = sum(1 for n in move_imgs if n.startswith("b_"))
    print(f"移走的里 b_ 前缀: {b_moved} 张, 非 b_: {len(move_imgs) - b_moved} 张")
    print("=" * 58)

    if not APPLY:
        print("\n[预演模式] 未移动任何文件。示例(前15条将被移走的):")
        for n, keep in to_move[:15]:
            tag = " +标注" if has_label(n) else ""
            print(f"  移走 {n}{tag}   (保留 {keep})")
        print("\n确认无误后,加 --apply 真正执行。")
        return

    # 2) 真正移动
    os.makedirs(IMG_Q, exist_ok=True)
    os.makedirs(LBL_Q, exist_ok=True)
    rows = []
    for name, keep in to_move:
        src_i = os.path.join(IMG, name)
        if os.path.exists(src_i):
            shutil.move(src_i, os.path.join(IMG_Q, name))
        lbl_moved = ""
        if has_label(name):
            src_l = os.path.join(LBL, stem(name) + ".txt")
            if os.path.exists(src_l):
                shutil.move(src_l, os.path.join(LBL_Q, stem(name) + ".txt"))
                lbl_moved = stem(name) + ".txt"
        rows.append([name, lbl_moved, keep])

    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["移走的图片", "移走的标注", "保留的对应图片"])
        w.writerows(rows)

    print(f"\n完成!已移动 {len(rows)} 张冗余图到 {IMG_Q}")
    print(f"对应标注移到 {LBL_Q}")
    print(f"完整清单: {MANIFEST}")
    print("如需还原:把隔离文件夹里的文件移回即可。")


if __name__ == "__main__":
    main()
