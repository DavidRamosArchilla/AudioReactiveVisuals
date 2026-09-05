import torch
from diffusers import AutoencoderKL, ZImagePipeline
import numpy as np
import imageio
import gc
from tqdm import tqdm

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)

pipe.to("cuda")

vae = pipe.vae
transformer = pipe.transformer
scheduler = pipe.scheduler
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

@torch.no_grad()
def encode_image(img):
    x = vae.encode(img.unsqueeze(0).to("cuda") * 2 - 1).latent_dist.sample()
    return x

@torch.no_grad()
def decode_image(latent):
    return (vae.decode(latent).sample / 2 + 0.5).clamp(0, 1)

@torch.no_grad()
def encode_prompt(prompt):
    """Encode text prompt to get caption features"""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to("cuda")
    prompt_embeds = text_encoder(text_input_ids, return_dict=False)[0]
    return prompt_embeds

# Generate images and get their latents + prompts
prompt_A = "A beautiful cat"
prompt_B = "A beautiful dog"

with torch.no_grad():
    image_A = pipe(prompt=prompt_A, num_inference_steps=4, guidance_scale=0.0).images[0]
    image_B = pipe(prompt=prompt_B, num_inference_steps=4, guidance_scale=0.0).images[0]

# Save the images for reference
imageio.imsave("image_A.png", np.array(image_A))
imageio.imsave("image_B.png", np.array(image_B))

# Encode images to latents
image_A_tensor = torch.tensor(np.array(image_A), dtype=vae.dtype).permute(2,0,1) / 255.0
image_B_tensor = torch.tensor(np.array(image_B), dtype=vae.dtype).permute(2,0,1) / 255.0

latent_A = encode_image(image_A_tensor)
latent_B = encode_image(image_B_tensor)

# Encode prompts to caption features
cap_feats_A = encode_prompt(prompt_A)
cap_feats_B = encode_prompt(prompt_B)

# Move to CPU to save GPU memory, we'll move back when needed
latent_A_cpu = latent_A.cpu()
latent_B_cpu = latent_B.cpu()
cap_feats_A_cpu = cap_feats_A.cpu()
cap_feats_B_cpu = cap_feats_B.cpu()

# Clear GPU memory
del latent_A, latent_B, cap_feats_A, cap_feats_B
torch.cuda.empty_cache()
gc.collect()

def add_noise_flow_matching(latent, t, noise=None):
    """Add noise according to flow matching schedule."""
    if noise is None:
        noise = torch.randn_like(latent)
    
    t_normalized = t / scheduler.config.num_train_timesteps
    noisy_latent = t_normalized * latent + (1 - t_normalized) * noise
    return noisy_latent

@torch.no_grad()
def denoise_flow_matching(noisy_latent, cap_feats, num_inference_steps=10):
    """Denoise using flow matching transformer with caption features"""
    scheduler.set_timesteps(num_inference_steps, device="cuda")
    
    latent = noisy_latent
    
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising", leave=False)):
        # Add temporal dimension
        if latent.dim() == 4:
            latent_with_time = latent.unsqueeze(2)
        else:
            latent_with_time = latent
        
        # Prepare lists
        latent_list = [latent_with_time[i] for i in range(latent_with_time.shape[0])]
        cap_feats_list = [cap_feats[i] for i in range(cap_feats.shape[0])]
        
        timestep = t.unsqueeze(0).expand(latent.shape[0])
        
        # Predict
        model_output = transformer(
            x=latent_list,
            t=timestep,
            cap_feats=cap_feats_list,
            return_dict=False,
        )[0]
        
        # Convert back to tensor
        if isinstance(model_output, list):
            model_output = torch.stack(model_output, dim=0)
        
        # Remove temporal dimension
        if model_output.dim() == 5 and model_output.shape[2] == 1:
            model_output = model_output.squeeze(2)
        
        # Step
        latent = scheduler.step(model_output, t, latent, return_dict=False)[0]
        
        # Clear intermediate tensors
        del model_output, latent_list, cap_feats_list, latent_with_time
        if i % 3 == 0:  # Periodic cleanup
            torch.cuda.empty_cache()
    
    return latent

def interpolate_with_flow_matching(latent_A, latent_B, cap_feats_A, cap_feats_B, 
                                   t_flow=600, lam=0.5, num_inference_steps=10):
    """Interpolate between two latents using flow matching"""
    # Move to GPU
    latent_A = latent_A.to("cuda")
    latent_B = latent_B.to("cuda")
    cap_feats_A = cap_feats_A.to("cuda")
    cap_feats_B = cap_feats_B.to("cuda")
    
    # Use the same noise for both
    noise = torch.randn_like(latent_A)
    
    # Add noise
    noisy_A = add_noise_flow_matching(latent_A, t_flow, noise)
    noisy_B = add_noise_flow_matching(latent_B, t_flow, noise)
    
    # Interpolate
    noisy_interp = (1 - lam) * noisy_A + lam * noisy_B
    cap_feats_interp = (1 - lam) * cap_feats_A + lam * cap_feats_B
    
    # Clean up before denoising
    del noisy_A, noisy_B, noise, latent_A, latent_B, cap_feats_A, cap_feats_B
    torch.cuda.empty_cache()
    
    # Denoise
    clean_interp = denoise_flow_matching(noisy_interp, cap_feats_interp, num_inference_steps)
    
    return clean_interp

frames = []
steps = 30

# Parameters
t_flow = 600
num_inference_steps = 4  # Reduced from 10 - model was trained for 4 steps

print("Generating interpolation frames with flow matching...")
for i in tqdm(range(steps+1), desc="Interpolation progress"):
    lam = i / steps
    
    # Interpolate
    latent_interp = interpolate_with_flow_matching(
        latent_A_cpu, 
        latent_B_cpu,
        cap_feats_A_cpu,
        cap_feats_B_cpu,
        t_flow=t_flow, 
        lam=lam,
        num_inference_steps=num_inference_steps
    )
    
    # Decode
    frame = decode_image(latent_interp)[0].permute(1,2,0).cpu().float().detach().numpy()
    frames.append((frame*255).astype(np.uint8))
    
    # Clean up
    del latent_interp
    torch.cuda.empty_cache()
    gc.collect()

imageio.mimsave("interp_flow_matching.gif", frames, fps=15)
print("Done! Saved to interp_flow_matching.gif")