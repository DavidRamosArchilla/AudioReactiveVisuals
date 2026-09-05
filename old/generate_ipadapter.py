import torch
import librosa
import numpy as np
import cv2
import os
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from moviepy.editor import ImageSequenceClip, AudioFileClip

# ==========================================
# 1. CONFIGURATION
# ==========================================
AUDIO_FILE = "input.mp3"       # Your song
INIT_IMAGE = "init.png"        # Starting image
OUTPUT_VIDEO = "output.mp4"
PROMPT = "bioluminescent jellyfish floating in deep space, nebula, stars, 8k, masterpiece, vibrant colors"
NEGATIVE_PROMPT = "blur, low quality, distortion, watermark, text, ugly, deformed"
FPS = 24
DEVICE = "cuda"

# Beat Sensitivity Settings
KICK_FREQ_THRESHOLD = 150.0  # Hz (Below this avg freq = Kick/Bass)
STRENGTH_KICK = 0.80         # High change for kicks (0.0 to 1.0)
STRENGTH_CLAP = 0.35         # Subtle change for claps/high-hats
SILENCE_THRESHOLD = 0.02     # RMS amplitude below this = silence

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_audio_data(audio_path):
    print(f"Loading audio: {audio_path}...")
    y, sr = librosa.load(audio_path)
    
    # Separate harmonic and percussive components for better beat tracking
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Detect Beat Frames using the percussive component
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Ensure start (0.0) and end of song are included in beat_times
    if beat_times.size == 0 or beat_times[0] > 0:
        beat_times = np.insert(beat_times, 0, 0.0)
    duration = librosa.get_duration(y=y, sr=sr)
    if beat_times[-1] < duration:
        beat_times = np.append(beat_times, duration)
        
    return y, sr, beat_times

def analyze_segment(y, sr, start_time, end_time):
    """
    Analyzes audio segment between beats to classify intensity and type.
    """
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Boundary safety
    if end_sample > len(y): end_sample = len(y)
    if start_sample >= end_sample: return "silence", 0.0

    segment = y[start_sample:end_sample]
    rms = np.sqrt(np.mean(segment**2))
    
    if rms < SILENCE_THRESHOLD:
        return "silence", 0.0
    
    # Spectral Centroid: Low value = Bass/Kick. High value = Treble/Clap.
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    avg_freq = np.mean(centroid)
    
    if avg_freq < KICK_FREQ_THRESHOLD:
        return "kick", STRENGTH_KICK
    else:
        return "clap", STRENGTH_CLAP

def slerp(val, low, high):
    """
    Spherical Linear Interpolation for PyTorch tensors.
    """
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

def interpolate_embeddings(embeds_A, embeds_B, alpha):
    """
    Interpolates between two IPAdapter embeddings (positive and negative).
    """
    img_emb_A, neg_img_emb_A = embeds_A
    img_emb_B, neg_img_emb_B = embeds_B
    
    mixed_img = slerp(alpha, img_emb_A, img_emb_B)
    mixed_neg = slerp(alpha, neg_img_emb_A, neg_img_emb_B)
    
    return (mixed_img, mixed_neg)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================

def main():
    # --- A. INITIALIZE MODELS ---
    print("Loading SDXL and IPAdapter... (This may take a moment)")
    
    # Load SDXL Image-to-Image Pipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(DEVICE)
    
    # Load IPAdapter
    pipe.load_ip_adapter(
        "h94/IP-Adapter", 
        subfolder="sdxl_models", 
        weight_name="ip-adapter_sdxl.bin"
    )
    pipe.set_ip_adapter_scale(0.6) # Scale of 0.6 is a good balance for morphing

    # --- B. PREPARE DATA ---
    y, sr, beat_times = load_audio_data(AUDIO_FILE)
    
    if os.path.exists(INIT_IMAGE):
        current_image = Image.open(INIT_IMAGE).convert("RGB").resize((1024, 1024))
    else:
        print(f"Error: {INIT_IMAGE} not found. Please provide a starting image.")
        return

    print("--- Phase 1: Generating Keyframes (The Beats) ---")
    keyframes = []
    keyframe_types = [] 
    
    # Save the very first frame
    keyframes.append(current_image)
    keyframe_types.append("start")
    
    # Iterate through beats to generate Anchor Images
    for i in range(len(beat_times) - 1):
        start = beat_times[i]
        end = beat_times[i+1]
        
        beat_type, strength = analyze_segment(y, sr, start, end)
        print(f"Processing Beat {i+1}/{len(beat_times)-1}: {beat_type} (Strength: {strength})")
        
        if beat_type == "silence":
            # For silence, generate a black frame to fade into
            next_image = Image.new('RGB', (1024, 1024), (0, 0, 0))
        else:
            # Generate new keyframe using img2img
            # High strength = new image differs significantly from previous
            next_image = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=current_image,
                strength=strength, 
                num_inference_steps=30,
                guidance_scale=7.5,
                ip_adapter_image=current_image # Condition on previous to keep thematic consistency
            ).images[0]
        
        keyframes.append(next_image)
        keyframe_types.append(beat_type)
        current_image = next_image

    print(f"Generated {len(keyframes)} keyframes. Preparing for Interpolation...")

    # --- C. PRE-CALCULATE EMBEDDINGS ---
    # We extract embeddings for all keyframes now to avoid re-calculating them inside the loop
    
    def get_embeds(pil_image):
        with torch.no_grad():
             # We use the pipeline's internal helper to get embeddings
             # Note: This returns a tuple (image_embeds, negative_image_embeds)
             embeds = pipe.prepare_ip_adapter_image_embeds(
                 ip_adapter_image=pil_image,
                 ip_adapter_image_embeds=None,
                 device=DEVICE,
                 num_images_per_prompt=1,
                 do_classifier_free_guidance=True
             )
        return embeds

    keyframe_embeds = [get_embeds(k) for k in keyframes]

    # --- D. PHASE 2: INTERPOLATION LOOP ---
    print("--- Phase 2: Rendering Video Frames ---")
    final_frames = []
    
    # Fixed generator ensures the background noise doesn't "boil" chaotically between beats
    fixed_generator = torch.Generator(device=DEVICE).manual_seed(42)

    for i in range(len(beat_times) - 1):
        start_t = beat_times[i]
        end_t = beat_times[i+1]
        duration = end_t - start_t
        num_frames = int(duration * FPS)
        
        if num_frames < 1: continue

        img_A = keyframes[i]
        img_B = keyframes[i+1]
        embeds_A = keyframe_embeds[i]
        embeds_B = keyframe_embeds[i+1]
        target_beat_type = keyframe_types[i+1]

        # Interpolate frames
        for f in range(num_frames):
            alpha = f / num_frames # Progress 0.0 -> 1.0
            
            # 1. Handle Silence (Simple Crossfade)
            if target_beat_type == "silence":
                 blended = Image.blend(img_A, img_B, alpha)
                 final_frames.append(np.array(blended))
                 continue
            
            # 2. Handle Active Beats (IPAdapter Morphing)
            # Mix the embeddings
            mixed_embeds = interpolate_embeddings(embeds_A, embeds_B, alpha)
            
            # Input image for continuity:
            # We use the previously generated frame (final_frames[-1]) if available.
            # This creates a "feedback loop" resulting in fluid motion.
            if len(final_frames) > 0:
                input_frame = Image.fromarray(final_frames[-1])
            else:
                input_frame = img_A

            # Generate frame
            # strength is LOW (0.35) because we only want to nudge the pixels slightly
            # toward the new embedding concept, preserving the motion from the feedback loop.
            frame_gen = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=input_frame,
                strength=0.35, 
                ip_adapter_image_embeds=mixed_embeds, 
                generator=fixed_generator, # IMPORTANT: Keep seed fixed
                num_inference_steps=20,    # Lower steps for speed in interpolation
                guidance_scale=7.5
            ).images[0]
            
            final_frames.append(np.array(frame_gen))

    # --- E. EXPORT ---
    print("Saving video to file...")
    clip = ImageSequenceClip(final_frames, fps=FPS)
    audio = AudioFileClip(AUDIO_FILE)
    
    # Sync audio length
    if audio.duration > clip.duration:
        audio = audio.subclip(0, clip.duration)
    clip = clip.set_audio(audio)
    
    clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac")
    print(f"Done! Video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()