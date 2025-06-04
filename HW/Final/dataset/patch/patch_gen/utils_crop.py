# ================================================================
# Iris Recognition • Eye-Patch version  (no rubber-sheet, no irisSeg)
# ------------------------------------------------
# 文件內包含所有 *.py 與一支 bash script；複製後依檔名拆開即可。
# 依賴：mediapipe >= 0.10.5、opencv-python、torch、torchvision、tqdm
# ================================================================

# ────────────────────────────────────────────────────────────────
# utils_crop.py
# ────────────────────────────────────────────────────────────────
"""Crop 224×224 eye patch.  
流程：
1. 嘗試 Mediapipe FaceMesh+iris landmark；取 5 點質心為中心，radius ≈ iris size。
2. 若 Mediapipe 失敗 → 回退圖像中心裁 224×224。
"""
import cv2, numpy as np
from typing import Union
CENTER_PATCH_SIZE = 224

# try import mediapipe
try:
    import mediapipe as mp
    _mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5)
    _MP_READY = True
except Exception:
    _MP_READY = False

_IRIS_IDS = [468, 469, 470, 471, 472]  # right-eye iris landmark ids


def _mp_crop(img: np.ndarray) -> Union[np.ndarray, None]:
    """Return crop or None if mediapipe fails."""
    results = _mp_face.process(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    if not results.multi_face_landmarks:
        return None
    lms = results.multi_face_landmarks[0].landmark
    pts = np.array([[lms[i].x, lms[i].y] for i in _IRIS_IDS])
    h, w = img.shape
    cx, cy = (pts.mean(axis=0) * [w, h]).astype(int)
    # half patch size；確保不越界
    hs = ws = CENTER_PATCH_SIZE // 2
    y1, y2 = np.clip(cy - hs, 0, h), np.clip(cy + hs, 0, h)
    x1, x2 = np.clip(cx - ws, 0, w), np.clip(cx + ws, 0, w)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < CENTER_PATCH_SIZE or crop.shape[1] < CENTER_PATCH_SIZE:
        return None
    return cv2.resize(crop, (CENTER_PATCH_SIZE, CENTER_PATCH_SIZE))


def crop_eye_patch(img_gray: np.ndarray) -> np.ndarray:
    if _MP_READY:
        patch = _mp_crop(img_gray)
        if patch is not None:
            return patch
    # fallback: center crop
    h, w = img_gray.shape
    hs = CENTER_PATCH_SIZE // 2
    cx, cy = w // 2, h // 2
    y1, y2 = cy - hs, cy + hs
    x1, x2 = cx - hs, cx + hs
    # pad if boundary
    pad_y1 = max(0, -y1); pad_x1 = max(0, -x1)
    pad_y2 = max(0, y2 - h); pad_x2 = max(0, x2 - w)
    img_padded = cv2.copyMakeBorder(img_gray, pad_y1, pad_y2, pad_x1, pad_x2,
                                    cv2.BORDER_REFLECT)
    y1 += pad_y1; y2 += pad_y1; x1 += pad_x1; x2 += pad_x1
    crop = img_padded[y1:y2, x1:x2]
    return cv2.resize(crop, (CENTER_PATCH_SIZE, CENTER_PATCH_SIZE))