import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)

prompts = [
    "Ethereal abstract fog of soft gradients, pastel colors slowly blending, low contrast, dreamlike diffusion",
    "Abstract geometric forms emerging from haze, muted neon edges, smooth interpolation, minimalist composition",
    "Flowing liquid shapes, iridescent surfaces, slow motion morphing, calm abstract energy",
    "Organic fractal patterns breathing outward, subtle glow, deep blues and violets, meditative abstraction",
    "Layered translucent planes intersecting, soft shadows, modern abstract spatial depth",
    "Particle clouds drifting and coalescing, gentle turbulence, cinematic abstract motion",
    "Abstract topographic waves, smooth contours, warm-to-cool gradient transition",
    "Surreal abstract light fields bending through invisible structures, minimal noise",
    "Slowly collapsing abstract symmetry, fluid geometry, pearlescent color palette",
    "Calm abstract convergence into a unified glowing form, soft focus, harmonious colors"
]

# pipe.transformer.compile()
pipe.to("cuda")
with torch.no_grad():
    for i, prompt in enumerate(prompts):
        print(f"Generating image for prompt {i+1}/{len(prompts)}")
        image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        # save the image
        image.save(f"generated_image_{i}.png")
