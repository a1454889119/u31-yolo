import random
import shutil
from pathlib import Path

src_dir = Path(r"D:\head_sole\data\dataset_2026_05_26\images")
dst_dir = Path(r"D:\head_sole\data\data_1000")

num_images = 1000
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

dst_dir.mkdir(parents=True, exist_ok=True)

images = [
    p for p in src_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_exts
]

selected = random.sample(images, min(num_images, len(images)))

for i, src_path in enumerate(selected, 1):
    dst_path = dst_dir / src_path.name

    if dst_path.exists():
        dst_path = dst_dir / f"calib_{i:04d}_{src_path.name}"

    shutil.copy2(src_path, dst_path)

print(f"copied {len(selected)} images to {dst_dir}")