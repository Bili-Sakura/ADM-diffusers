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
  --image-size 64 \
  --num-inference-steps 250 \
  --output samples_64.npz \
  --image-output-dir outputs
```

## Notes

- This refactor keeps legacy ADM internals as implementation dependencies.
- The public API now centers around `diffusers`-style classes and checkpoint folders.
