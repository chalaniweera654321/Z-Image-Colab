import os, random, time, shutil

import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS

# ── Model Loading ──────────────────────────────────────────────
print("\n" + "="*50)
print("  Z-Image-Turbo + LoRA Starting Up")
print("="*50)

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# LoRA loader
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()

startup_start = time.time()

with torch.inference_mode():

    print("\n[1/4] Loading UNet... ", end="", flush=True)
    t0 = time.time()

    base_unet = UNETLoader.load_unet(
        "z_image_turbo_bf16.safetensors",
        "default"
    )[0]

    print(f"done ({time.time()-t0:.1f}s)")

    print("[2/4] Loading CLIP (Qwen3)... ", end="", flush=True)
    t0 = time.time()

    base_clip = CLIPLoader.load_clip(
        "qwen_3_4b.safetensors",
        type="lumina2"
    )[0]

    print(f"done ({time.time()-t0:.1f}s)")

    print("[3/4] Loading VAE... ", end="", flush=True)
    t0 = time.time()

    vae = VAELoader.load_vae("ae.safetensors")[0]

    print(f"done ({time.time()-t0:.1f}s)")

print(f"\n✅ Base models loaded in {time.time()-startup_start:.1f}s")
print("="*50 + "\n")


# ── LoRA ───────────────────────────────────────────────────────
# Put your LoRA inside:
#
# ComfyUI/models/loras/
#
# Example:
# ComfyUI/models/loras/my_lora.safetensors

DEFAULT_LORA = "my_lora.safetensors"

def load_lora(lora_name, model_strength=1.0, clip_strength=1.0):

    if not lora_name:
        print("ℹ️ No LoRA selected")
        return base_unet, base_clip

    print("\n" + "="*50)
    print("  Loading LoRA")
    print("="*50)

    print(f"LoRA: {lora_name}")
    print(f"Model strength: {model_strength}")
    print(f"CLIP strength: {clip_strength}")

    t0 = time.time()

    model, clip = LoraLoader.load_lora(
        base_unet,
        base_clip,
        lora_name,
        float(model_strength),
        float(clip_strength)
    )

    print(f"✅ LoRA loaded ({time.time()-t0:.1f}s)")
    print("="*50 + "\n")

    return model, clip


# ── Helpers ────────────────────────────────────────────────────

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)


def get_save_path(prompt):

    safe_prompt = re.sub(
        r'[^a-zA-Z0-9_-]',
        '_',
        prompt
    )[:25]

    uid = uuid.uuid4().hex[:6]

    filename = f"{safe_prompt}_{uid}.png"

    return os.path.join(save_dir, filename)


# ── Generation ─────────────────────────────────────────────────

@torch.inference_mode()
def generate(input):

    values = input["input"]

    positive_prompt = values["positive_prompt"]
    negative_prompt = values["negative_prompt"]

    seed = values["seed"]
    steps = values["steps"]
    cfg = values["cfg"]

    sampler_name = values["sampler_name"]
    scheduler = values["scheduler"]

    denoise = values["denoise"]

    width = values["width"]
    height = values["height"]

    batch_size = values["batch_size"]

    # LoRA settings
    lora_name = values["lora_name"]
    lora_strength = values["lora_strength"]
    clip_strength = values["clip_strength"]

    print("\n" + "="*50)
    print("  New Generation Request")
    print("="*50)

    total_start = time.time()

    # ── Load LoRA ──────────────────────────────────────────────

    print("\n[1/5] Applying LoRA... ", end="", flush=True)

    t0 = time.time()

    model, clip = load_lora(
        lora_name,
        lora_strength,
        clip_strength
    )

    print(f"done ({time.time()-t0:.1f}s)")


    # ── Encode prompts ────────────────────────────────────────

    print("[2/5] Encoding prompts... ", end="", flush=True)

    t0 = time.time()

    positive = CLIPTextEncode.encode(
        clip,
        positive_prompt
    )[0]

    negative = CLIPTextEncode.encode(
        clip,
        negative_prompt
    )[0]

    print(f"done ({time.time()-t0:.1f}s)")


    # ── Latent ────────────────────────────────────────────────

    print("[3/5] Creating latent image... ", end="", flush=True)

    t0 = time.time()

    latent_image = EmptyLatentImage.generate(
        width,
        height,
        batch_size=batch_size
    )[0]

    print(f"done ({time.time()-t0:.1f}s)")


    # ── Sampling ──────────────────────────────────────────────

    print(
        f"[4/5] Sampling ({steps} steps, "
        f"LoRA strength={lora_strength})..."
    )

    t0 = time.time()

    samples = KSampler.sample(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise
    )[0]

    print(
        f"      Sampling done "
        f"({time.time()-t0:.1f}s)"
    )


    # ── Decode ────────────────────────────────────────────────

    print(
        "[5/5] Decoding image (VAE)... ",
        end="",
        flush=True
    )

    t0 = time.time()

    decoded = VAEDecode.decode(
        vae,
        samples
    )[0].detach()

    print(
        f"done ({time.time()-t0:.1f}s)"
    )


    # ── Save ──────────────────────────────────────────────────

    save_path = get_save_path(
        positive_prompt
    )

    Image.fromarray(
        np.array(
            decoded * 255,
            dtype=np.uint8
        )[0]
    ).save(save_path)

    print(
        f"\n💾 Saved to : {save_path}"
    )


    # ── Google Drive ─────────────────────────────────────────

    drive_path = "/content/gdrive/MyDrive/z_image_turbo"

    if os.path.exists(drive_path):

        shutil.copy(
            save_path,
            drive_path
        )

        print(
            f"☁️ Copied to Google Drive: "
            f"{drive_path}"
        )


    print(
        f"✅ Total: "
        f"{time.time()-total_start:.1f}s"
    )

    print("="*50 + "\n")

    return save_path, seed


# ── Find available LoRAs ───────────────────────────────────────

def get_lora_files():

    lora_dir = "./models/loras"

    if not os.path.exists(lora_dir):
        return []

    files = []

    for filename in os.listdir(lora_dir):

        if filename.lower().endswith(
            (".safetensors", ".pt", ".ckpt")
        ):
            files.append(filename)

    return sorted(files)


# ── Gradio UI ──────────────────────────────────────────────────

import gradio as gr


def generate_ui(
    positive_prompt,
    negative_prompt,
    width,
    height,
    seed,
    steps,
    batch_size,
    cfg,
    denoise,
    lora_name,
    lora_strength,
    clip_strength,
    sampler_name="euler",
    scheduler="simple"
):

    input_data = {
        "input": {

            "positive_prompt":
                positive_prompt,

            "negative_prompt":
                negative_prompt,

            "width":
                int(width),

            "height":
                int(height),

            "batch_size":
                int(batch_size),

            "seed":
                int(seed),

            "steps":
                int(steps),

            "cfg":
                float(cfg),

            "sampler_name":
                sampler_name,

            "scheduler":
                scheduler,

            "denoise":
                float(denoise),

            "lora_name":
                lora_name,

            "lora_strength":
                float(lora_strength),

            "clip_strength":
                float(clip_strength),
        }
    }

    image_path, used_seed = generate(
        input_data
    )

    return (
        image_path,
        image_path,
        used_seed
    )


DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp. She has almond shaped red eyes and she's holding a intricate mask. She's wearing white and gold royal gown with a black cloak. In the veins of her neck its gold."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus, pixelated"""


custom_css = """
.gradio-container {
    font-family:
        'SF Pro Display',
        -apple-system,
        BlinkMacSystemFont,
        sans-serif;
}
"""


lora_files = get_lora_files()

if not lora_files:
    lora_files = [""]


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:

    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column;
align-items:center; justify-content:center; margin:20px 0;">

    <h1 style="font-size:2.5em; margin-bottom:10px;">
        Z-Image-Turbo + LoRA
    </h1>

</div>
""")


    with gr.Row():

        with gr.Column():

            positive = gr.Textbox(
                DEFAULT_POSITIVE,
                label="Positive Prompt",
                lines=5
            )


            with gr.Row():

                width = gr.Number(
                    value=1024,
                    label="Width",
                    precision=0
                )

                height = gr.Number(
                    value=1024,
                    label="Height",
                    precision=0
                )

                seed = gr.Number(
                    value=0,
                    label="Seed (0 = random)",
                    precision=0
                )

                steps = gr.Slider(
                    4,
                    25,
                    value=9,
                    step=1,
                    label="Steps"
                )

                batch_size = gr.Number(
                    value=1,
                    label="Batch Size",
                    precision=0
                )


            # ── LoRA Settings ───────────────────────────────

            with gr.Accordion(
                "🎨 LoRA Settings",
                open=True
            ):

                lora_name = gr.Dropdown(
                    choices=lora_files,
                    value=lora_files[0],
                    label="LoRA"
                )

                with gr.Row():

                    lora_strength = gr.Slider(
                        -2.0,
                        2.0,
                        value=1.0,
                        step=0.05,
                        label="LoRA Model Strength"
                    )

                    clip_strength = gr.Slider(
                        -2.0,
                        2.0,
                        value=1.0,
                        step=0.05,
                        label="LoRA CLIP Strength"
                    )


            with gr.Row():

                run = gr.Button(
                    "🚀 Generate",
                    variant="primary"
                )


            with gr.Accordion(
                "Image Settings",
                open=False
            ):

                with gr.Row():

                    cfg = gr.Slider(
                        0.5,
                        4.0,
                        value=1.0,
                        step=0.1,
                        label="CFG"
                    )

                    denoise = gr.Slider(
                        0.1,
                        1.0,
                        value=1.0,
                        step=0.05,
                        label="Denoise"
                    )


                with gr.Row():

                    negative = gr.Textbox(
                        DEFAULT_NEGATIVE,
                        label="Negative Prompt",
                        lines=3
                    )


        with gr.Column():

            download_image = gr.File(
                label="Download Image"
            )

            output_img = gr.Image(
                label="Generated Image",
                height=480
            )

            used_seed = gr.Textbox(
                label="Seed Used",
                interactive=False
            )


    run.click(
        fn=generate_ui,

        inputs=[
            positive,
            negative,
            width,
            height,
            seed,
            steps,
            batch_size,
            cfg,
            denoise,
            lora_name,
            lora_strength,
            clip_strength
        ],

        outputs=[
            download_image,
            output_img,
            used_seed
        ]
    )


demo.launch(
    share=True,
    debug=True
)
