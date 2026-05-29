import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diffusers import DDIMScheduler
from diffusers.pipelines.adm.pipeline_adm import ADMPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from an ADM diffusers pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Path to converted pipeline directory")
    parser.add_argument("--output", type=str, default="samples.npz", help="Output .npz path")
    parser.add_argument("--image-output-dir", type=str, default="", help="Optional folder to dump PNG images")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Classifier guidance; use 0 for UNet-only")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    pipeline = ADMPipeline.from_pretrained(args.model).to(args.device)
    if args.use_ddim:
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    all_samples = []
    image_dir = Path(args.image_output_dir) if args.image_output_dir else None
    if image_dir:
        image_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    while produced < args.num_samples:
        current_bs = min(args.batch_size, args.num_samples - produced)
        class_labels = None
        if args.class_label is not None:
            class_labels = [args.class_label] * current_bs

        result = pipeline(
            batch_size=current_bs,
            class_labels=class_labels,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_type="np",
        )
        batch = np.stack(result.images, axis=0)
        batch = (batch * 255.0).round().astype(np.uint8)
        all_samples.append(batch)

        if image_dir:
            for i in range(batch.shape[0]):
                Image.fromarray(batch[i]).save(image_dir / f"sample_{produced + i:06d}.png")
        produced += current_bs
        print(f"Generated {produced}/{args.num_samples}")

    arr = np.concatenate(all_samples, axis=0)[: args.num_samples]
    np.savez(args.output, arr=arr)
    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    main()
