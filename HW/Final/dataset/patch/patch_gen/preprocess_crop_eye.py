# ────────────────────────────────────────────────────────────────
# preprocess_crop_eye.py
# ────────────────────────────────────────────────────────────────
import argparse, pathlib, tqdm, cv2, os
from utils_crop import crop_eye_patch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out_root = pathlib.Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    paths = list(pathlib.Path(args.root).rglob('*.jpg'))
    ok = mp_fail = 0
    for p in tqdm.tqdm(paths, desc='crop'):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            mp_fail += 1; continue
        patch = crop_eye_patch(img)
        rel = p.relative_to(args.root).with_suffix('.jpg')
        (out_root/rel.parent).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_root/rel), patch)
        ok += 1
    print(f'Finished. success={ok}  skipped={len(paths)-ok}')