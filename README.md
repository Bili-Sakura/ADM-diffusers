# ADM Diffusers Refactor

This repository was refactored into a `huggingface/diffusers`-style layout inspired by NiT-diffusers:

- `src/diffusers/models/unets/unet_adm.py`: `ADMUNet2DModel`
- `src/diffusers/schedulers/scheduling_adm.py`: `ADMScheduler`
- `src/diffusers/pipelines/adm/pipeline_adm.py`: `ADMPipeline`
- `scripts/convert_adm_to_diffusers.py`: convert legacy ADM checkpoints
- `scripts/sample_adm.py`: sample from converted pipelines

Legacy train/sample entry scripts were removed in favor of a model/scheduler/pipeline organization.

## Convert a legacy checkpoint

```bash
python scripts/convert_adm_to_diffusers.py \
  --checkpoint models/64x64_diffusion.pt \
  --output adm-64-diffusers \
  --image-size 64 \
  --num-channels 192 \
  --num-res-blocks 3 \
  --attention-resolutions 32,16,8 \
  --learn-sigma \
  --class-cond
```

## Sample images

```bash
python scripts/sample_adm.py \
  --model adm-64-diffusers \
  --num-samples 100 \
  --batch-size 4 \
  --num-inference-steps 250 \
  --guidance-scale 0 \
  --use-ddim \
  --class-label 207 \
  --output samples_64.npz \
  --image-output-dir outputs
```

## ADM-G sampling notes

- UNet outputs **6 channels** when `learn_sigma=True` (3 for noise ε, 3 for learned variance σ).
- `guidance_scale` is **classifier guidance** strength (ADM-G), not Stable-Diffusion-style CFG.
- **UNet-only** sampling (no classifier): `guidance_scale=0` with `class_labels` still required for class-conditional checkpoints.
- With **DDIM**, the pipeline splits ε/σ before `scheduler.step`; with **DDPM + `learned_range`**, all 6 channels are kept.

```python
from diffusers import DDIMScheduler

pipe = ADMPipeline.from_pretrained("path/to/checkpoint").to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
class_id = pipe.get_label_ids("golden retriever")[0]

# UNet-only (no classifier guidance)
image = pipe(class_labels=class_id, guidance_scale=0.0, num_inference_steps=50).images[0]

# Full ADM-G
image = pipe(class_labels=class_id, guidance_scale=4.0, num_inference_steps=50).images[0]
```

## Notes

- This refactor keeps legacy ADM internals as implementation dependencies.
- The public API now centers around `diffusers`-style classes and checkpoint folders.
