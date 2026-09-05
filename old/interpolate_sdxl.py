import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL
import numpy as np
import imageio
from tqdm import tqdm

# Load SDXL Turbo
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Generate images
with torch.no_grad():
    image_A = pipe(
        prompt="A beautiful cat", 
        num_inference_steps=1, 
        guidance_scale=0.0
    ).images[0]
    image_B = pipe(
        prompt="A beautiful dog", 
        num_inference_steps=1, 
        guidance_scale=0.0
    ).images[0]

# Save reference images
imageio.imsave("image_A.png", np.array(image_A))
imageio.imsave("image_B.png", np.array(image_B))

vae = pipe.vae
scheduler = pipe.scheduler
unet = pipe.unet

@torch.no_grad()
def encode(img):
    """Encode image to latent space"""
    x = vae.encode(img.unsqueeze(0).to("cuda") * 2 - 1).latent_dist.sample()
    return x * vae.config.scaling_factor

@torch.no_grad()
def decode(latent):
    """Decode latent to image space"""
    latent = latent / vae.config.scaling_factor
    return (vae.decode(latent).sample / 2 + 0.5).clamp(0, 1)

def add_noise(latent, timestep, noise=None):
    """Add noise to latent according to diffusion schedule"""
    if noise is None:
        noise = torch.randn_like(latent)
    
    # Get noise schedule values
    sqrt_alpha_prod = scheduler.alphas_cumprod[timestep] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - scheduler.alphas_cumprod[timestep]) ** 0.5
    
    noisy_latent = sqrt_alpha_prod * latent + sqrt_one_minus_alpha_prod * noise
    return noisy_latent

@torch.no_grad()
def denoise_latent(noisy_latent, start_timestep, num_inference_steps=20):
    """Denoise a latent starting from a specific timestep"""
    # Set up timesteps starting from start_timestep
    scheduler.set_timesteps(num_inference_steps, device="cuda")
    
    # Get text embeddings (empty prompt for unconditional)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        "",
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )
    
    # SDXL requires these additional embeddings
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = pipe._get_add_time_ids(
        original_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        target_size=(1024, 1024),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    )
    add_time_ids = add_time_ids.to("cuda")
    
    # Find the closest timestep index to our start_timestep
    timesteps = scheduler.timesteps
    start_idx = (timesteps <= start_timestep).nonzero(as_tuple=True)[0]
    if len(start_idx) == 0:
        start_idx = 0
    else:
        start_idx = start_idx[0].item()
    
    latent = noisy_latent
    
    # Denoise from start_timestep to 0
    for i, t in enumerate(tqdm(timesteps[start_idx:], desc="Denoising", leave=False)):
        # Prepare added conditioning
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids
        }
        
        # Predict noise
        noise_pred = unet(
            latent,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        
        # Compute previous latent
        latent = scheduler.step(noise_pred, t, latent, return_dict=False)[0]
    
    return latent

def interpolate_with_diffusion(latent_A, latent_B, t_noise=500, lam=0.5, num_inference_steps=20):
    """
    Interpolate between two latents using the diffusion process:
    1. Add noise to both latents (forward process to timestep t)
    2. Linearly interpolate in the noisy space
    3. Denoise back to clean latent (reverse process)
    """
    # Use the same noise for both to keep them aligned
    noise = torch.randn_like(latent_A)
    
    # Add noise to both latents (forward diffusion process)
    noisy_A = add_noise(latent_A, t_noise, noise)
    noisy_B = add_noise(latent_B, t_noise, noise)
    
    # Interpolate in noisy space
    noisy_interp = (1 - lam) * noisy_A + lam * noisy_B
    
    # Denoise back (reverse diffusion process)
    clean_interp = denoise_latent(noisy_interp, t_noise, num_inference_steps)
    
    return clean_interp

# Encode images to latent space
image_A_tensor = torch.tensor(np.array(image_A), dtype=torch.float16).permute(2,0,1) / 255.0
image_B_tensor = torch.tensor(np.array(image_B), dtype=torch.float16).permute(2,0,1) / 255.0

latent_A = encode(image_A_tensor)
latent_B = encode(image_B_tensor)

frames = []
steps = 30

# Interpolation parameters
t_noise = 500  # Noise level (0-999): higher = more creative interpolation
num_inference_steps = 20  # Denoising steps: more = better quality

print("Generating interpolation frames...")
for i in tqdm(range(steps+1), desc="Overall progress"):
    lam = i / steps
    
    # Interpolate using diffusion process
    latent_interp = interpolate_with_diffusion(
        latent_A, 
        latent_B, 
        t_noise=t_noise, 
        lam=lam,
        num_inference_steps=num_inference_steps
    )
    
    # Decode to image
    frame = decode(latent_interp)[0].permute(1,2,0).cpu().float().numpy()
    frames.append((frame*255).astype(np.uint8))

imageio.mimsave("interp_sdxl_diffusion.gif", frames, fps=15)
print("Done! Saved to interp_sdxl_diffusion.gif")