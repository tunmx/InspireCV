#!/usr/bin/env python3
import os
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except Exception as e:
    sys.stderr.write("This script requires opencv-python and numpy. Install via:\n")
    sys.stderr.write("  pip install opencv-python numpy\n")
    raise


def get_images_dir() -> Path:
    env = os.getenv("INSPIRECV_IMAGES_DIR")
    if env and len(env) > 0:
        return Path(env)
    # default relative to repo root
    return Path("images")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_png(path: Path, img):
    ensure_dir(path.parent)
    if not cv2.imwrite(str(path), img):
        raise RuntimeError(f"Failed to write {path}")


def main():
    sizes = [128, 256, 512, 1024]
    images_dir = get_images_dir()
    src_path = images_dir / "kun.jpg"
    source_mode = os.getenv("INSPIRECV_GT_SOURCE", "kun").lower()  # 'kun' or 'synthetic'
    if source_mode == "kun" and not src_path.exists():
        raise FileNotFoundError(f"Input image not found: {src_path} (set INSPIRECV_IMAGES_DIR or use INSPIRECV_GT_SOURCE=synthetic)")
    out_dir = images_dir / "image_gt"
    ensure_dir(out_dir)

    # Read as BGR when using kun
    if source_mode == "kun":
        src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if src is None:
            raise RuntimeError(f"cv2.imread failed: {src_path}")
    else:
        # Synthetic base will be created per-size, so src is unused
        src = None

    for N in sizes:
        if source_mode == "kun":
            # Resize (nearest and linear) from kun.jpg
            r_near = cv2.resize(src, (N, N), interpolation=cv2.INTER_NEAREST)
            r_lin = cv2.resize(src, (N, N), interpolation=cv2.INTER_LINEAR)
        else:
            # Build a clean synthetic base pattern (B=gradX, G=gradY, R=checker)
            gx = np.tile(np.linspace(0, 255, N, dtype=np.uint8), (N, 1))
            gy = np.tile(np.linspace(0, 255, N, dtype=np.uint8)[:, None], (1, N))
            tile = max(4, N // 16)
            yy, xx = np.indices((N, N))
            checker = (((xx // tile + yy // tile) % 2) * 255).astype(np.uint8)
            r_lin = cv2.merge([gx, gy, checker])  # B,G,R
            r_near = r_lin.copy()
        save_png(out_dir / f"resize_nearest_{N}.png", r_near)
        save_png(out_dir / f"resize_linear_{N}.png", r_lin)

        # Derive geometry from the linear base (stable for further ops)
        base = r_lin

        # Rotations
        rot90 = cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)
        rot180 = cv2.rotate(base, cv2.ROTATE_180)
        rot270 = cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE)
        save_png(out_dir / f"rot90_{N}.png", rot90)
        save_png(out_dir / f"rot180_{N}.png", rot180)
        save_png(out_dir / f"rot270_{N}.png", rot270)

        # Flips
        flip_h = cv2.flip(base, 1)
        flip_v = cv2.flip(base, 0)
        save_png(out_dir / f"flip_h_{N}.png", flip_h)
        save_png(out_dir / f"flip_v_{N}.png", flip_v)

        # ---------- Section 5: Color/Channel/Pixel ops GT ----------
        # ToGray (OpenCV BGR->GRAY, matches library weights)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        save_png(out_dir / f"gray_{N}.png", gray)

        # SwapRB
        swaprb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        save_png(out_dir / f"swaprb_{N}.png", swaprb)

        # MeanChannels: floor((B+G+R)/3)
        mean = (base.astype(np.uint16).sum(axis=2) // 3).astype(np.uint8)
        save_png(out_dir / f"mean_{N}.png", mean)

        # AbsDiff: base vs flip_h
        absdiff = cv2.absdiff(base, flip_h)
        save_png(out_dir / f"absdiff_flip_h_{N}.png", absdiff)

        # Blend with horizontal mask gradient and a pure green image as the other image
        mask = np.tile(np.linspace(0, 255, N, dtype=np.uint8), (N, 1))
        save_png(out_dir / f"mask_grad_{N}.png", mask)
        m01 = (mask.astype(np.float32) * (1.0 / 255.0))[:, :, None]
        green = np.zeros_like(base, dtype=np.uint8)
        green[:, :, 1] = 255  # BGR: G=255
        save_png(out_dir / f"green_{N}.png", green)
        blend_g = (m01 * base.astype(np.float32) + (1.0 - m01) * green.astype(np.float32)).round().clip(0, 255).astype(np.uint8)
        save_png(out_dir / f"blend_maskgrad_green_{N}.png", blend_g)

        # Threshold on gray (binary, >100 -> 255)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        save_png(out_dir / f"thresh100_gray_{N}.png", thresh)

        # ---------- Section 6: Filtering / Morphology GT ----------
        # GaussianBlur 5x5 (sigma auto) with replicate border
        g_bgr = cv2.GaussianBlur(base, (5, 5), 0, borderType=cv2.BORDER_REPLICATE)
        save_png(out_dir / f"gauss5_bgr_{N}.png", g_bgr)
        g_gray = cv2.GaussianBlur(gray, (5, 5), 0, borderType=cv2.BORDER_REPLICATE)
        save_png(out_dir / f"gauss5_gray_{N}.png", g_gray)

        # Erode/Dilate 3x3 single-iteration on gray with replicate border
        k3 = np.ones((3, 3), dtype=np.uint8)
        er = cv2.erode(gray, k3, iterations=1, borderType=cv2.BORDER_REPLICATE)
        di = cv2.dilate(gray, k3, iterations=1, borderType=cv2.BORDER_REPLICATE)
        save_png(out_dir / f"erode3_gray_{N}.png", er)
        save_png(out_dir / f"dilate3_gray_{N}.png", di)

        # Crop: center region (N/4, N/4, N/2, N/2)
        rx, ry = N // 4, N // 4
        rw, rh = N // 2, N // 2
        crop = base[ry:ry + rh, rx:rx + rw, :]
        save_png(out_dir / f"crop_{N}_x{rx}_y{ry}_w{rw}_h{rh}.png", crop)

        # Pad constant 10 on all sides (black)
        t = b = l = r = 10
        pad = cv2.copyMakeBorder(base, t, b, l, r, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        save_png(out_dir / f"pad_{N}_t{t}_b{b}_l{l}_r{r}_black.png", pad)

        # WarpAffine 180° using cv2 (matches rotation but via affine)
        # Use pixel-centered coordinates: ((N-1)/2, (N-1)/2) to align with dst->src matrix
        # semantics used in C++ WarpAffine tests and avoid half-pixel drift.
        center = ((N - 1) * 0.5, (N - 1) * 0.5)
        Msd = cv2.getRotationMatrix2D(center, 180.0, 1.0)  # src->dst
        warp180 = cv2.warpAffine(base, Msd, (N, N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        save_png(out_dir / f"warp_rot180_{N}.png", warp180)

    print(f"GT images generated under: {out_dir}")


if __name__ == "__main__":
    main()


