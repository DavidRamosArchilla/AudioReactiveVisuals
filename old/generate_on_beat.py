import librosa
import torch
import numpy as np
from diffusers import AutoPipelineForText2Image
from PIL import Image
from moviepy.editor import ImageSequenceClip
from moviepy.editor import AudioFileClip

# 1. SETUP
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
pipe = AutoPipelineForText2Image.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
audio_path = "audios/SefaHarderclass.wav"

# 2. AUDIO ANALYSIS
y, sr = librosa.load(audio_path)
# Get the "RMS" (Loudness) over time
rms = librosa.feature.rms(y=y)[0]
# Get the beat frames (indexes where beats happen)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

def slerp(val, low, high):
    """
    Spherical Linear Interpolation between two tensors (low and high).
    val: float between 0.0 and 1.0 (the progress)
    """
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

def interpolate_images(pipe, img_a, img_b, num_frames=12):
    """
    Returns a list of PIL images morphing from img_a to img_b.
    """
    # 1. PRE-PROCESS IMAGES FOR VAE
    # The VAE expects tensors normalized between -1 and 1
    def preprocess(image):
        w, h = image.size
        # Resize to ensuring it matches model expectation (usually multiples of 8)
        w, h = map(lambda x: x - x % 8, (w, h)) 
        image = image.resize((w, h), resample=Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    tensor_a = preprocess(img_a).to(pipe.device).half() # use .half() if using float16
    tensor_b = preprocess(img_b).to(pipe.device).half()

    # 2. ENCODE IMAGES TO LATENTS
    # We use the pipe's VAE (Variational Autoencoder)
    with torch.no_grad():
        latent_a = pipe.vae.encode(tensor_a).latent_dist.sample()
        latent_b = pipe.vae.encode(tensor_b).latent_dist.sample()
        
        # Scaling is required for SD 1.5/2.1 latents
        latent_a = latent_a * pipe.vae.config.scaling_factor
        latent_b = latent_b * pipe.vae.config.scaling_factor

    # 3. LOOP AND SLERP
    interpolated_images = []
    
    # Generate ratios (e.g., 0.0, 0.1, 0.2 ... 1.0)
    ratios = np.linspace(0, 1, num_frames)
    
    with torch.no_grad():
        for ratio in ratios:
            # Interpolate the latents
            # We assume batch size is 1, so we squeeze/unsqueeze as needed for the slerp func
            lat_interp = slerp(float(ratio), latent_a, latent_b)
            
            # Decode back to image
            lat_interp = lat_interp / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(lat_interp).sample
            
            # Post-process (Tensor to PIL)
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).round().astype("uint8")
            interpolated_images.append(Image.fromarray(image[0]))
            
    return interpolated_images


# 3. THE LOOP
# dummy image to start with
dummy_image = Image.new("RGB", (512, 512), color="white")
current_image = pipe(
    prompt="A futuristic city, neon lights",
    image=dummy_image, # will be ignored due to strength=1
    num_inference_steps=9,
    guidance_scale=0.0,
    strength=1, # Full generation
).images[0]
all_video_frames = [current_image]

print("Starting Generation Loop...")

for i, beat_time in enumerate(beat_times):
    
    # Get loudness at this specific beat moment
    # (We map the beat index to the rms array index)
    rms_index = librosa.time_to_frames(beat_time, sr=sr)
    loudness = rms[rms_index] # Value roughly between 0.0 and 1.0 (normalize if needed)
    
    # Map loudness to "Change Amount" (Strength)
    # If loud (0.8), change image a lot. If quiet (0.1), change little.
    change_strength = max(0.2, min(0.8, loudness))
    
    # Generate the target image for this beat
    next_image = pipe(
        prompt="A futuristic city", 
        image=current_image, 
        strength=change_strength,
        guidance_scale=0,
        num_inference_steps=9
    ).images[0]
    
    # Calculate how many frames needed between these beats
    # (Distance in seconds * FPS)
    duration = beat_times[i] - (beat_times[i-1] if i > 0 else 0)
    fps = 24
    frames_count = int(duration * fps)
    
    # Ensure at least 1 frame to avoid errors
    frames_count = max(1, frames_count)
    
    # Create the morph sequence
    morph_sequence = interpolate_images(pipe, current_image, next_image, num_frames=frames_count)
    
    # Add to master list
    all_video_frames.extend(morph_sequence)
    
    # Update current image for next iteration
    current_image = next_image

# 4. EXPORT VIDEO USING MOVIEPY
print("Saving Video...")
# Convert PIL images to numpy arrays for MoviePy if necessary, 
# but ImageSequenceClip usually handles paths or arrays.
# It's safest to convert PIL -> Numpy array

numpy_frames = [np.array(img) for img in all_video_frames]
clip = ImageSequenceClip(numpy_frames, fps=24)

# Load the original audio and trim it to the video length
audio = AudioFileClip("audio_path")
final_clip = clip.set_audio(audio.set_duration(clip.duration))

final_clip.write_videofile("output_reactive.mp4", codec="libx264")

