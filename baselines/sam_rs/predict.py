"""A thin SAM (Segment Anything) wrapper for remote sensing segmentation.

SAM (Kirillov et al., ICCV 2023) is prompt-driven: you give points / boxes and
it returns masks. RS images are huge, so the key engineering is the
**preprocessing + coordinate transform** (resize longest side to 1024, scale
prompts accordingly). This wrapper makes that explicit and reusable.

Quick check (no checkpoint, no GPU, no segment-anything install needed)::

    python predict.py --demo

Real inference (needs the `segment-anything` package + a checkpoint)::

    pip install git+https://github.com/facebookresearch/segment-anything.git
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

    # box prompt
    python predict.py --image scene.tif --checkpoint sam_vit_h_4b8939.pth \
        --box 150 100 300 200

    # point prompt(s): x y [x y ...]
    python predict.py --image scene.tif --checkpoint sam_vit_h_4b8939.pth \
        --point 225 150

    # everything (automatic mask generation)
    python predict.py --image scene.tif --checkpoint sam_vit_h_4b8939.pth --auto
"""

from __future__ import annotations

import argparse
import os

import numpy as np

TARGET_SIZE = 1024  # SAM encoder input resolution (longest side)


# --------------------------------------------------------------------------- #
# Preprocessing / prompt utilities (no SAM install required)
# --------------------------------------------------------------------------- #
def resize_longest_side(image: np.ndarray, target: int = TARGET_SIZE):
    """Nearest-neighbour resize so the longest side == target. Returns (img, scale)."""
    h, w = image.shape[:2]
    scale = target / float(max(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    ys = (np.arange(new_h) / scale).astype(np.int64).clip(0, h - 1)
    xs = (np.arange(new_w) / scale).astype(np.int64).clip(0, w - 1)
    return image[ys][:, xs], scale


def transform_points(points, scale):
    """Scale (x, y) point prompts from original to resized coordinates."""
    return [(round(float(x) * scale, 2), round(float(y) * scale, 2)) for x, y in points]


def transform_box(box, scale):
    """Scale an xyxy box prompt from original to resized coordinates."""
    return [round(float(v) * scale, 2) for v in box]


# --------------------------------------------------------------------------- #
# SAM wrapper
# --------------------------------------------------------------------------- #
class SAMRSPredictor:
    """Lazy SAM wrapper. Heavily biased toward 'show me the pipeline clearly'."""

    def __init__(self, checkpoint=None, model_type="vit_h", device="cpu",
                 points_per_side=32) -> None:
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self.points_per_side = points_per_side
        self._predictor = None

    def _ensure(self):
        if self._predictor is not None:
            return self._predictor
        from segment_anything import SamPredictor, sam_model_registry  # lazy import
        if not self.checkpoint or not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                "SAM checkpoint not found. Download e.g.\n"
                "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                "and pass it via --checkpoint."
            )
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        self._predictor = SamPredictor(sam)
        return self._predictor

    def set_image(self, image: np.ndarray) -> None:
        self._ensure().set_image(image)

    def predict_points(self, points, labels):
        masks, scores, _ = self._ensure().predict(
            point_coords=np.array(points), point_labels=np.array(labels),
            multimask_output=True,
        )
        return masks, scores

    def predict_box(self, box):
        masks, scores, _ = self._ensure().predict(
            box=np.array(box), multimask_output=False)
        return masks, scores

    def auto_mask(self, image: np.ndarray):
        from segment_anything import SamAutomaticMaskGenerator
        gen = SamAutomaticMaskGenerator(self._ensure().model,
                                        points_per_side=self.points_per_side)
        return gen.generate(image)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _synthetic_image(h: int = 384, w: int = 512) -> np.ndarray:
    g = np.random.default_rng(0)
    img = (g.random((h, w, 3)) * 60 + 40).astype(np.uint8)
    img[100:200, 150:300] = 220  # a bright "building" rectangle
    return img


def demo() -> None:
    print("== SAM-RS pipeline demo (no checkpoint required) ==")
    img = _synthetic_image()
    print("input image        :", img.shape, img.dtype)

    resized, scale = resize_longest_side(img)
    print(f"resized (longest={TARGET_SIZE}) :", resized.shape, f"(scale={scale:.3f})")

    pts = [(225, 150)]  # (x, y) inside the bright rectangle
    print("point prompt orig  :", pts)
    print("point prompt scaled:", transform_points(pts, scale))

    box = [150, 100, 300, 200]  # xyxy
    print("box prompt orig    :", box)
    print("box prompt scaled  :", transform_box(box, scale))

    try:
        import segment_anything  # noqa: F401
        print("[ok] `segment-anything` installed -> add --checkpoint to get real masks")
    except ImportError:
        print("[info] `segment-anything` not installed. To run real masks:")
        print("       pip install git+https://github.com/facebookresearch/segment-anything.git")
        print("       wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("       python predict.py --image scene.tif --checkpoint sam_vit_h_4b8939.pth --point 225 150")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="SAM wrapper for RS segmentation")
    ap.add_argument("--demo", action="store_true", help="offline pipeline check (no weights)")
    ap.add_argument("--image", type=str, help="input RS image (png/tif)")
    ap.add_argument("--checkpoint", type=str, default=None, help="SAM .pth checkpoint")
    ap.add_argument("--model-type", type=str, default="vit_h",
                    choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--point", type=float, nargs="+", help="point prompt: x y [x y ...]")
    ap.add_argument("--box", type=float, nargs=4, help="box prompt: x1 y1 x2 y2")
    ap.add_argument("--auto", action="store_true", help="automatic mask generation")
    ap.add_argument("--out", type=str, default="mask.npy")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.demo or not args.image:
        demo()
        return

    from PIL import Image
    img = np.asarray(Image.open(args.image).convert("RGB"))
    pred = SAMRSPredictor(checkpoint=args.checkpoint, model_type=args.model_type,
                          device=args.device)

    if args.auto:
        masks = pred.auto_mask(img)
        np.save(args.out, np.array([m["segmentation"] for m in masks]))
        print(f"auto: {len(masks)} masks -> {args.out}")
    elif args.box:
        masks, scores = pred.predict_box(args.box)
        np.save(args.out, masks[0])
        print(f"box prompt -> {args.out}  (score={float(scores[0]):.3f})")
    elif args.point:
        pts = [(args.point[i], args.point[i + 1]) for i in range(0, len(args.point), 2)]
        masks, scores = pred.predict_points(pts, [1] * len(pts))
        best = int(np.argmax(scores))
        np.save(args.out, masks[best])
        print(f"point prompt -> {args.out}  (score={float(scores[best]):.3f})")
    else:
        ap.error("provide --point, --box, or --auto (or use --demo)")


if __name__ == "__main__":
    main()
