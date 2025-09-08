# overlay_coco_segmentation.py
import cv2
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import os

# ===== 固定設定 =====
ANN_PATH = "annotations/5images_3classes.json"
IMG_DIR = Path("support_images")
OUT_DIR = Path("output_overlay")
ALPHA = 0.5       # 透明度
DRAW_EDGE = True  # 輪郭も描画するかどうか
# ====================

def color_from_id(cat_id: int) -> tuple:
    np.random.seed(cat_id)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))

def main():
    coco = COCO(ANN_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cat_id2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    img_ids = coco.getImgIds()

    for img_id in img_ids:
        img_obj = coco.loadImgs([img_id])[0]
        img_path = IMG_DIR / img_obj["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"読み込み失敗: {img_path}")
            continue
        H, W = image.shape[:2]

        overlay = image.copy()
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))

        for ann in anns:
            cat_id = ann["category_id"]
            color = color_from_id(cat_id)

            mask = coco.annToMask(ann)
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

            colored = np.zeros_like(image)
            colored[:] = color
            overlay[mask.astype(bool)] = cv2.addWeighted(
                colored[mask.astype(bool)], ALPHA, overlay[mask.astype(bool)], 1 - ALPHA, 0
            )

            if DRAW_EDGE:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)

        out_path = OUT_DIR / f"{Path(img_obj['file_name']).stem}_overlay.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"保存: {out_path}")

if __name__ == "__main__":
    main()


