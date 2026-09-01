#!/usr/bin/env python3
"""Create a degraded copy of an image-classification dataset by Gaussian-blurring
a fraction of the images in every leaf folder, keeping the rest untouched.

Used to build a controlled "some support images are low quality" scenario for
comparing QA-ProtoNet against a standard Prototypical Network, since the source
ice-ice photos are mostly sharp.

Example:
    python scripts/degrade_dataset.py \
        --src "dataset ice ice" --dst "dataset ice ice blur50" --ratio 0.5
"""

import argparse
import json
import os
import random
import shutil

from PIL import Image, ImageFilter

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

def blur_radius(img, frac_min, frac_max, rng):
    frac = rng.uniform(frac_min, frac_max)
    return max(1.0, frac * min(img.size))

def save_image(img, dst_path):
    ext = os.path.splitext(dst_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        img.save(dst_path, quality=95)
    else:
        img.save(dst_path)

def process_dir(src_root, dst_root, ratio, frac_min, frac_max, rng):
    manifest = []
    for dirpath, _dirnames, filenames in os.walk(src_root):
        rel_dir = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        image_files = sorted(f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
        other_files = [f for f in filenames if f not in image_files and not f.startswith('.')]

        for f in other_files:
            shutil.copy2(os.path.join(dirpath, f), os.path.join(dst_dir, f))

        n_blur = round(len(image_files) * ratio)
        blurred_names = set(rng.sample(image_files, n_blur)) if image_files else set()

        for f in image_files:
            src_path = os.path.join(dirpath, f)
            dst_path = os.path.join(dst_dir, f)
            rel_path = os.path.join(rel_dir, f).replace(os.sep, '/')

            img = Image.open(src_path).convert('RGB')

            if f in blurred_names:
                radius = blur_radius(img, frac_min, frac_max, rng)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                manifest.append({'path': rel_path, 'blurred': True, 'blur_radius': round(radius, 2)})
            else:
                manifest.append({'path': rel_path, 'blurred': False, 'blur_radius': 0.0})

            save_image(img, dst_path)

    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', default='dataset ice ice',
                         help="source dataset root (default: 'dataset ice ice')")
    parser.add_argument('--dst', default='dataset ice ice blur50',
                         help="output dataset root (default: 'dataset ice ice blur50')")
    parser.add_argument('--ratio', type=float, default=0.5,
                         help="fraction of images to blur, applied within every leaf folder so "
                              "each class/split keeps the same ratio (default: 0.5)")
    parser.add_argument('--frac-min', type=float, default=0.006,
                         help="minimum Gaussian blur radius, as a fraction of each image's "
                              "shorter side (default: 0.006)")
    parser.add_argument('--frac-max', type=float, default=0.018,
                         help="maximum Gaussian blur radius, as a fraction of each image's "
                              "shorter side (default: 0.018)")
    parser.add_argument('--seed', type=int, default=42,
                         help="random seed, for a reproducible choice of which images get "
                              "blurred and how strongly (default: 42)")
    args = parser.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise ValueError("--dst must be different from --src")

    rng = random.Random(args.seed)
    manifest = process_dir(args.src, args.dst, args.ratio, args.frac_min, args.frac_max, rng)

    manifest_path = os.path.join(args.dst, 'blur_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump({
            'src': args.src,
            'ratio': args.ratio,
            'frac_min': args.frac_min,
            'frac_max': args.frac_max,
            'seed': args.seed,
            'images': manifest,
        }, f, indent=2)

    n_blurred = sum(1 for m in manifest if m['blurred'])
    print("Wrote {:d} images to '{:s}' ({:d} blurred, {:d} clean)".format(
        len(manifest), args.dst, n_blurred, len(manifest) - n_blurred))
    print("Manifest: {:s}".format(manifest_path))

if __name__ == '__main__':
    main()
