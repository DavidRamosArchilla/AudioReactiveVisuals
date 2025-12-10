import torch
from diffusers import AutoencoderKL, ZImagePipeline
import numpy as np
import imageio
import gc

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
# pipe.transformer.compile()
pipe.to("cuda")
with torch.no_grad():
    image_A = pipe(prompt="A beautiful cat", num_inference_steps=4, guidance_scale=0.0).images[0]
    image_B = pipe(prompt="A beautiful dog", num_inference_steps=4, guidance_scale=0.0).images[0]

torch.cuda.empty_cache()
del pipe
gc.collect()

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
vae.to("cuda")

@torch.no_grad()
def encode(img):
    x = vae.encode(img.unsqueeze(0).to("cuda") * 2 - 1).latent_dist.sample()
    return x

@torch.no_grad()
def decode(latent):
    return (vae.decode(latent).sample / 2 + 0.5).clamp(0, 1)

def slerp(t, v0, v1):
    v0_u = v0 / torch.norm(v0)
    v1_u = v1 / torch.norm(v1)
    dot = (v0_u * v1_u).sum()
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return torch.sin((1.0-t)*omega)/so * v0 + torch.sin(t*omega)/so * v1

# Load preprocessed tensors image_A, image_B
image_A = torch.tensor(np.array(image_A), dtype=vae.dtype).permute(2,0,1) / 255.0
image_B = torch.tensor(np.array(image_B), dtype=vae.dtype).permute(2,0,1) / 255.0

latent_A = encode(image_A)
latent_B = encode(image_B)

frames = []
steps = 30

for i in range(steps+1):
    t = i / steps
    latent = slerp(t, latent_A, latent_B)
    frame = decode(latent)[0].permute(1,2,0).cpu().detach().numpy()
    frames.append((frame*255).astype(np.uint8))

imageio.mimsave("interp.gif", frames, fps=15)
