import os, random, time, shutil

import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS

# ── Model Loading ──────────────────────────────────────────────
print("\n" + "="*50)
print("  Z-Image-Turbo Starting Up")
print("="*50)

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

startup_start = time.time()

with torch.inference_mode():
    print("\n[1/3] Loading UNet... ", end="", flush=True)
    t0 = time.time()
    unet = UNETLoader.load_unet("z_image_turbo_bf16.safetensors", "default")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[2/3] Loading CLIP (Qwen3)... ", end="", flush=True)
    t0 = time.time()
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[3/3] Loading VAE... ", end="", flush=True)
    t0 = time.time()
    vae = VAELoader.load_vae("ae.safetensors")[0]
    print(f"done ({time.time()-t0:.1f}s)")

print(f"\n✅ All models loaded in {time.time()-startup_start:.1f}s")
print("="*50 + "\n")

# ── Helpers ────────────────────────────────────────────────────
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt):
    safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
    uid = uuid.uuid4().hex[:6]
    filename = f"{safe_prompt}_{uid}.png"
    return os.path.join(save_dir, filename)

# ── Generation ─────────────────────────────────────────────────
@torch.inference_mode()
def generate(input):
    values = input["input"]
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']
    denoise = values['denoise']
    width = values['width']
    height = values['height']
    batch_size = values['batch_size']

    print("\n" + "="*50)
    print("  New Generation Request")
    print("="*50)

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
        print(f"  Seed       : {seed} (randomized)")
    else:
        print(f"  Seed       : {seed}")

    print(f"  Size       : {width}x{height}")
    print(f"  Steps      : {steps}")
    print(f"  CFG        : {cfg}")
    print(f"  Sampler    : {sampler_name} / {scheduler}")
    print(f"  Denoise    : {denoise}")
    print(f"  Prompt     : {positive_prompt[:80]}{'...' if len(positive_prompt) > 80 else ''}")
    print("="*50)

    total_start = time.time()

    print("\n[1/4] Encoding prompts... ", end="", flush=True)
    t0 = time.time()
    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[2/4] Creating latent image... ", end="", flush=True)
    t0 = time.time()
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print(f"[3/4] Sampling ({steps} steps)...")
    t0 = time.time()
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    print(f"      Sampling done ({time.time()-t0:.1f}s)")

    print("[4/4] Decoding image (VAE)... ", end="", flush=True)
    t0 = time.time()
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    print(f"done ({time.time()-t0:.1f}s)")

    save_path = get_save_path(positive_prompt)
    Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(save_path)
    print(f"\n💾 Saved to : {save_path}")

    drive_path = "/content/gdrive/MyDrive/z_image_turbo"
    if os.path.exists(drive_path):
        shutil.copy(save_path, drive_path)
        print(f"☁️  Copied to Google Drive: {drive_path}")

    print(f"✅ Total    : {time.time()-total_start:.1f}s")
    print("="*50 + "\n")

    return save_path, seed


# ── Gradio UI ──────────────────────────────────────────────────
import gradio as gr

def generate_ui(
    positive_prompt,
    negative_prompt,
    width,
    height,
    seed,
    steps,
    cfg,
    denoise,
    batch_size=1,
    sampler_name="euler",
    scheduler="simple"
):
    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
        }
    }

    image_path, seed = generate(input_data)
    return image_path, image_path, seed


def update_dims(aspect):
    w, h = [int(x) for x in aspect.split("(")[0].strip().split("x")]
    return w, h


DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp. She has almond shaped red eyes and she's holding a intricate mask. She's wearing white and gold royal gown with a black cloak.  In the veins of her neck its gold."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus, pixelated"""

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]

custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks() as demo:
    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
    <h1 style="font-size:2.5em; margin-bottom:10px;">Z-Image-Turbo</h1>
    <a href="https://github.com/Tongyi-MAI/Z-Image" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white"
             style="height:15px;">
    </a>
</div>
""")

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)
            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
            with gr.Row():
                width = gr.Number(value=1024, label="Width", precision=0)
                height = gr.Number(value=1024, label="Height", precision=0)
            with gr.Row():
                run = gr.Button('🚀 Generate', variant='primary')
            with gr.Accordion('Image Settings', open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)

        with gr.Column():
            download_image = gr.File(label="Download Image")
            output_img = gr.Image(label="Generated Image", height=480)
            used_seed = gr.Textbox(label="Seed Used", interactive=False)

    aspect.change(fn=update_dims, inputs=aspect, outputs=[width, height])

    run.click(
        fn=generate_ui,
        inputs=[positive, negative, width, height, seed, steps, cfg, denoise],
        outputs=[download_image, output_img, used_seed]
    )

demo.launch(theme=gr.themes.Soft(), css=custom_css, share=True, debug=True)
