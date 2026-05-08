from typing import Optional

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ...models import ADMUNet2DModel
from ...schedulers import ADMScheduler


class ADMPipeline(DiffusionPipeline):
    def __init__(self, unet: ADMUNet2DModel, scheduler: ADMScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        image_size: Optional[int] = None,
        num_inference_steps: int = 250,
        use_ddim: bool = False,
        clip_denoised: bool = True,
        class_labels: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        del generator  # API placeholder for compatibility.
        self.unet.eval()
        runtime = self.scheduler.create_runtime(num_inference_steps=num_inference_steps, use_ddim=use_ddim)
        device = next(self.unet.parameters()).device
        if image_size is None:
            image_size = int(self.unet.config.image_size)
        model_kwargs = {}
        if class_labels is not None:
            model_kwargs["y"] = class_labels.to(device)
        sample_fn = runtime.ddim_sample_loop if use_ddim else runtime.p_sample_loop
        return sample_fn(
            self.unet.model,
            (batch_size, 3, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
        )
