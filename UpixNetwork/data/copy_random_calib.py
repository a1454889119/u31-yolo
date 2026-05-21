import os
import random
import shutil
from pathlib import Path

src_dir = Path(r"C:\沙包\datasets7500\images\train")
dst_dir = Path(r"C:\u31-yolo\UpixNetwork\data\shabao_calib")

num_images = 1000

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

dst_dir.mkdir(parents=True, exist_ok=True)

images = [
    p for p in src_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_exts
]

print(f"found {len(images)} images")

if len(images) == 0:
    raise RuntimeError(f"no images found in {src_dir}")

sample_count = min(num_images, len(images))
selected = random.sample(images, sample_count)

for i, src_path in enumerate(selected, 1):
    dst_path = dst_dir / src_path.name

    # 如果目标目录里已有同名文件，自动加前缀避免覆盖
    if dst_path.exists():
        dst_path = dst_dir / f"calib_{i:04d}_{src_path.name}"

    shutil.copy2(src_path, dst_path)

print(f"copied {sample_count} images to {dst_dir}")