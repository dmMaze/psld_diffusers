from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import deprecate

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AsymmetricAutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionPipelineOutput, retrieve_timesteps
from diffusers import DDIMScheduler

from .measurements import get_operator, get_noise


def psld_gaussian_noise(sigma=.05):
    return get_noise(name='gaussian', sigma=sigma)


def freeze_module(module: nn.Module):
    module = module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


class StableDiffusionInpaintPSLDPipeline(StableDiffusionInpaintPipeline):

    # @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = '',
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        gamma: float = 1e-2,
        omega: float = 1e-1,
        omit_jacobian: bool = False,
        enable_psld: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 999,
        timesteps: List[int] = None,
        guidance_scale: float = 1.,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        num_optimize_steps: int = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be inpainted (which parts of the image to
                be masked out with `mask_image` and repainted according to `prompt`). For both numpy array and pytorch
                tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the
                expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the
                expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but
                if passing latents directly it is not encoded again.
            mask_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            padding_mask_crop (`int`, *optional*, defaults to `None`):
                The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to
                image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region
                with the same aspect ration of the image and contains all masked area, and then expand that area based
                on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before
                resizing to the original image size for inpainting. This is useful when the masked area is small while
                the image is large and contain information irrelevant for inpainting, such as background.
            strength (`float`, *optional*, defaults to 1.0):
                Not used.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInpaintPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

        >>> init_image = download_image(img_url).resize((512, 512))
        >>> mask_image = download_image(mask_url).resize((512, 512))

        >>> pipe = StableDiffusionInpaintPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        >>> image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        if not isinstance(self.scheduler, DDIMScheduler):
            print(f'replace scheduler with ddim...')
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)

        with torch.no_grad():
            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )

            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs
            self.check_inputs(
                prompt,
                image,
                mask_image,
                height,
                width,
                strength,
                callback_steps,
                output_type,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                callback_on_step_end_tensor_inputs,
                padding_mask_crop,
            )

            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs
            self._interrupt = False

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                clip_skip=self.clip_skip,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )

            # 4. set timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps=num_inference_steps, strength=strength, device=device
            )
            # check that number of inference steps is not < 1 - as this doesn't make sense
            if num_inference_steps < 1:
                raise ValueError(
                    f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                    f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
                )
            # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
            is_strength_max = strength == 1.0

            # 5. Preprocess mask and image

            if padding_mask_crop is not None:
                crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
                resize_mode = "fill"
            else:
                crops_coords = None
                resize_mode = "default"

            original_image = image
            init_image = self.image_processor.preprocess(
                image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
            )
            init_image = init_image.to(dtype=torch.float32)

            # 6. Prepare latent variables
            num_channels_latents = self.vae.config.latent_channels
            num_channels_unet = self.unet.config.in_channels
            return_image_latents = num_channels_unet == 4

            latents_outputs = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                image=init_image,
                timestep=latent_timestep,
                is_strength_max=is_strength_max,
                return_noise=True,
                return_image_latents=return_image_latents,
            )

            if return_image_latents:
                latents, noise, image_latents = latents_outputs
            else:
                latents, noise = latents_outputs

            # 7. Prepare mask latent variables
            mask_condition = self.mask_processor.preprocess(
                mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
            )

            if masked_image_latents is None:
                masked_image = init_image * (mask_condition < 0.5)
                # masked_image = cv2.inpaint(init_image * (mask_condition < 0.5), mask_condition > 0.5, 3, cv2.INPAINT_NS)
            else:
                masked_image = masked_image_latents

            mask, masked_image_latents = self.prepare_mask_latents(
                mask_condition,
                masked_image,
                batch_size * num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                self.do_classifier_free_guidance,
            )

            # 8. Check that sizes of mask, masked image and latents match
            if num_channels_unet == 9:
                # default case for runwayml/stable-diffusion-inpainting
                num_channels_mask = mask.shape[1]
                num_channels_masked_image = masked_image_latents.shape[1]
                if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                    raise ValueError(
                        f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                        f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input."
                    )
            elif num_channels_unet != 4:
                raise ValueError(
                    f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
                )

            # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 9.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None
                else None
            )

            # 9.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            ip_mask = (1 - mask_condition).to(device=device, dtype=latents.dtype)
            if enable_psld:
                psld_noiser = psld_gaussian_noise()
                operator = get_operator(name='inpainting', device=device)
                y = operator.forward(init_image.to(device=device, dtype=latents.dtype), mask=ip_mask)
                psld_noise = torch.randn_like(y) * 0.05
                measurements = psld_noiser(y)
            else:
                measurements = self._encode_vae_image(init_image.to(device=device, dtype=latents.dtype) * ip_mask, generator=generator)
                ip_mask = torch.nn.functional.interpolate(ip_mask, (latents.shape[2], latents.shape[3]), mode='nearest')

            # 10. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            
            if num_optimize_steps is None:
                num_optimize_steps = num_inference_steps
            if enable_psld:
                num_optimize_steps = num_inference_steps

            area_not_inpaint = ip_mask.sum()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latents = torch.clone(latents.detach())
                latents.requires_grad = True

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                if omit_jacobian:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                else:
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                latents_prev = scheduler_output.prev_sample
                z0_pred = scheduler_output.pred_original_sample

                meas_error = inpaint_error = 0
                if enable_psld:
                    image_pred = self.vae.decode(
                            z0_pred / self.vae.config.scaling_factor, return_dict=False, generator=generator,
                        )[0]
                    meas_pred = operator.forward(image_pred,mask=ip_mask)
                    meas_pred = psld_noiser(meas_pred)
                    # meas_error = ((meas_pred - measurements) ** 2).mean()
                    meas_error = torch.linalg.norm(meas_pred - measurements)

                    ortho_project = image_pred - operator.transpose(operator.forward(image_pred, mask=ip_mask))
                    parallel_project = operator.transpose(measurements)
                    inpainted_image = parallel_project + ortho_project

                    encoded_z_0 = self._encode_vae_image(inpainted_image, generator=generator)
                    # inpaint_error = ((encoded_z_0 - z0_pred) ** 2).mean()
                    inpaint_error = torch.linalg.norm(encoded_z_0 - z0_pred)
                else:
                    if area_not_inpaint > 0:
                        meas_error = (((z0_pred - measurements) * ip_mask) ** 2).sum() / (area_not_inpaint * 3)

                error = inpaint_error * gamma + meas_error * omega
                if error > 0 and i <= num_optimize_steps:
                    gradients = torch.autograd.grad(error, inputs=latents)[0]
                    latents = latents_prev - gradients
                else:
                    latents = latents_prev

                if num_channels_unet == 4 and i > num_optimize_steps:
                    init_latents_proper = image_latents

                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    mask = callback_outputs.pop("mask", mask)
                    masked_image_latents = callback_outputs.pop("masked_image_latents", masked_image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        with torch.no_grad():
            if not output_type == "latent":
                condition_kwargs = {}
                if isinstance(self.vae, AsymmetricAutoencoderKL):
                    init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                    init_image_condition = init_image.clone()
                    init_image = self._encode_vae_image(init_image, generator=generator)
                    mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                    condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
                image = self.vae.decode(
                    latents / self.vae.config.scaling_factor, return_dict=False, generator=generator, **condition_kwargs
                )[0]
                has_nsfw_concept = None
                # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = latents
                has_nsfw_concept = None

            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            if padding_mask_crop is not None:
                image = [self.image_processor.apply_overlay(mask_image, original_image, i, crops_coords) for i in image]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


