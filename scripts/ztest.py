"""Kick test: can the full Z-Image (non-turbo) replace Turbo in the beat chain?
Times 1x txt2img + 2x chained img2img at 640x368 and extrapolates to a full song.
Run on GPU node: sbatch test_zimage.sh
"""
import time
import torch
from diffusers import ZImagePipeline, ZImageImg2ImgPipeline
from PIL import Image
import numpy as np

MODEL = "Tongyi-MAI/Z-Image"
W, H = 640, 368
STEPS = 30
GUID = 4.0
PROMPT = ("painterly matte painting, ancient greek marble temple on sunlit cliff, "
          "pine trees, blue sky, white clouds, soft sunlight, strong contrast")
NEG = "blurry, low detail, watermark, text, night"

print(f"[test] cuda={torch.cuda.get_device_name(0)}", flush=True)
t = time.time()
pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                      low_cpu_mem_usage=False)
pipe.to("cuda")
print(f"[test] txt2img pipe load: {time.time()-t:.1f}s", flush=True)

g = torch.Generator("cuda").manual_seed(7)
t = time.time()
img = pipe(prompt=PROMPT, negative_prompt=NEG, height=H, width=W,
           num_inference_steps=STEPS, guidance_scale=GUID,
           generator=g).images[0]
dt_t2i = time.time() - t
img.save("output/ztest_t2i.png")
print(f"[test] txt2img {STEPS} steps: {dt_t2i:.1f}s", flush=True)
del pipe
torch.cuda.empty_cache()

t = time.time()
i2i = ZImageImg2ImgPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                            low_cpu_mem_usage=False)
i2i.to("cuda")
print(f"[test] img2img pipe load: {time.time()-t:.1f}s", flush=True)
prev = img
dts = []
for i in range(2):
    g = torch.Generator("cuda").manual_seed(100 + i)
    t = time.time()
    prev = i2i(prompt=PROMPT, negative_prompt=NEG, image=prev,
               strength=0.4, num_inference_steps=STEPS, guidance_scale=GUID,
               generator=g).images[0]
    dts.append(time.time() - t)
    prev.save(f"output/ztest_i2i{i}.png")
    print(f"[test] img2img {i} {STEPS} steps: {dts[-1]:.1f}s", flush=True)

# Extrapolate: NewBeginning ~188.6s @200BPM -> ~628 beats (1 t2i + 627 i2i)
n = 627
total = dt_t2i + n * float(np.mean(dts))
print(f"[test] MEAN img2img: {np.mean(dts):.1f}s -> full song estimate: {total/60:.0f} min", flush=True)
print("[test] UNDER_2H" if total < 7200 else "[test] OVER_2H", flush=True)
