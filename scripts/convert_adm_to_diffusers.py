import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from diffusers.models.unets.unet_adm import ADMUNet2DModel
from diffusers.pipelines.adm.pipeline_adm import ADMPipeline
from diffusers.schedulers.scheduling_adm import ADMScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert legacy ADM checkpoint to diffusers-style directory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to legacy model .pt checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-resolutions", type=str, default="16,8")
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--class-cond", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", type=str, default="linear")
    return parser


def main():
    args = build_parser().parse_args()
    unet = ADMUNet2DModel(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        learn_sigma=args.learn_sigma,
        class_cond=args.class_cond,
        use_fp16=args.use_fp16,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    unet.load_state_dict(state_dict, strict=True)

    scheduler = ADMScheduler(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
    )
    pipeline = ADMPipeline(unet=unet, scheduler=scheduler)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(str(out_dir))
    print(f"Saved converted pipeline to {out_dir}")


if __name__ == "__main__":
    main()
