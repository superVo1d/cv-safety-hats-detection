import random
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "datasets" / "hardhat"

VAL_RATIO = 0.2
SEED = 42

CLASS_NAMES = {
    0: "no_helmet",
    1: "helmet"
}


def collect_pairs(images_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img_path in images_dir.glob("*.png"):
        label_path = img_path.with_suffix(".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def split_pairs(pairs: list[tuple[Path, Path]]):
    pairs = pairs.copy()
    random.Random(SEED).shuffle(pairs)
    val_count = max(1, int(len(pairs) * VAL_RATIO)) if len(pairs) > 1 else 0
    return pairs[val_count:], pairs[:val_count]


def copy_split(pairs: list[tuple[Path, Path]], output_dir: Path, split_name: str):
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for img_path, label_path in pairs:
        shutil.copy2(img_path, images_out / img_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in CLASS_NAMES.items())
    content = (
        f"path: .\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    pairs = collect_pairs(IMAGES_DIR)
    if not pairs:
        raise SystemExit(f"Не достаточно пар image+txt в {IMAGES_DIR}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    train_pairs, val_pairs = split_pairs(pairs)
    copy_split(train_pairs, OUTPUT_DIR, "train")
    copy_split(val_pairs, OUTPUT_DIR, "val")
    write_data_yaml(OUTPUT_DIR)

    print(f"train: {len(train_pairs)}, val: {len(val_pairs)}")
    print(f"data.yaml: {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
