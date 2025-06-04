import argparse, random, itertools
from pathlib import Path
from typing import List, Tuple
import os


# --------------------------------------------------------------------------- #
def scan_dataset(root: str, exclude_range: Tuple[int, int] = None):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for img in Path(root).rglob("*"):
        if img.suffix.lower() not in exts:
            continue
        parts = img.parts
        if len(parts) < 3:
            continue
        eye_id = parts[-3]  # 000, 001 ...
        side = parts[-2].upper()  # L / R
        if side not in ("L", "R"):
            continue
        # 排除指定範圍的 eye_id
        if exclude_range is not None:
            try:
                eye_num = int(eye_id)
                if exclude_range[0] <= eye_num <= exclude_range[1]:
                    continue
            except ValueError:
                pass
        paths.append((str(img), eye_id))
    return paths


def generate_pairs(
    records: List[Tuple[str, str]], num_pairs: int, seed: int = 0
) -> List[Tuple[str, str, int]]:
    random.seed(seed)
    by_id = {}
    for p, eid in records:
        by_id.setdefault(eid, []).append(p)
    ids = list(by_id.keys())

    if len(ids) < 2:
        raise RuntimeError("eye-id < 2")

    pos_all = []
    for eid, lst in by_id.items():
        if len(lst) < 2:
            continue
        for p1, p2 in itertools.combinations(lst, 2):
            pos_all.append((p1, p2, 1))

    neg_all = []
    for eid1, eid2 in itertools.combinations(ids, 2):
        for p1 in by_id[eid1]:
            for p2 in by_id[eid2]:
                neg_all.append((p1, p2, 0))

    need_pos = need_neg = num_pairs // 2
    pos = (
        random.sample(pos_all, need_pos)
        if need_pos <= len(pos_all)
        else random.choices(pos_all, k=need_pos)
    )
    neg = (
        random.sample(neg_all, need_neg)
        if need_neg <= len(neg_all)
        else random.choices(neg_all, k=need_neg)
    )

    pairs = pos + neg
    random.shuffle(pairs)
    return pairs


def write_pairs_txt(pairs: List[Tuple[str, str, int]], out_path: str):
    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w") as f:
        for p1, p2, lbl in pairs:
            p1 = p1.replace("origin/", "")
            p2 = p2.replace("origin/", "")
            f.write(f"{p1},{p2},{lbl}\n")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="Iris dataset root (e.g. ./dataset/CASIA-Iris-Thousand)",
    )
    parser.add_argument("--num_pairs", type=int, default=40000)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="validation ratio")
    parser.add_argument("--out_dir", default="./train_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_range", type=str, default=None,
                        help="Range of subjects to exclude, e.g. '1,80' to exclude 001 to 080")

    args = parser.parse_args()

    exclude_range = None
    if args.exclude_range:
        try:
            low, high = map(int, args.exclude_range.split(","))
            exclude_range = (low, high)
        except Exception as e:
            raise ValueError(f"Invalid exclude_range format: {args.exclude_range}") from e

    recs = scan_dataset(args.root, exclude_range=exclude_range)
    n_img = len(recs)
    n_id = len(set(eid for _, eid in recs))
    print(f"scan {n_img} images, totally {n_id} eye-id")

    pairs = generate_pairs(recs, args.num_pairs, seed=args.seed)
    split = int(len(pairs) * (1 - args.val_ratio))
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    train_txt = Path(args.out_dir) / "train_pairs.txt"
    val_txt = Path(args.out_dir) / "val_pairs.txt"
    write_pairs_txt(train_pairs, train_txt)
    write_pairs_txt(val_pairs, val_txt)

    print(f"train_pairs: {len(train_pairs)}  val_pairs: {len(val_pairs)}")
    print(f"train_pairs output to: {train_txt}")
    print(f"val_pairs output to: {val_txt}")