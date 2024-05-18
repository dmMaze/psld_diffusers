PSLD inpainting for diffuser pipeline.


## Example

``` python
from psld.inpainting_pipeline import StableDiffusionInpaintPSLDPipeline
import torch
import numpy as np
from PIL import Image
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

pretrained_p = 'runwayml/stable-diffusion-v1-5'
pipeline = StableDiffusionInpaintPSLDPipeline.from_pretrained(
    pretrained_p,
    torch_dtype=torch.float32,
    use_safetensors=True,
    requires_safety_checker=False,
)
pipeline = pipeline.to("cuda")

gamma = 1e-2
omega = 1e-1
prompt = ''
init_image = Image.open('workspace/inpaint_exp.png').convert('RGB')
mask = Image.open('workspace/inpaint_exp_mask.png').convert('L')

new_image = pipeline(prompt=prompt, image=init_image, mask_image=mask, enable_psld=True, gamma=gamma, omega=omega).images[0]
new_image.save('local_tst.jpg')
```

* Only inpainting is implemented
* Reduce num_inference_steps and scale gamma & omega accordingly could (not always) achieve similar results.
* Requires at least 18 GB VRAM for 512x512

## Reference
https://github.com/LituRout/PSLD