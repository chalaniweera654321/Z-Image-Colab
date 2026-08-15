import os
import random
import time
import shutil
import re
import uuid

import torch
import numpy as np
from PIL import Image

from nodes import NODE_CLASS_MAPPINGS


# ============================================================
# Z-IMAGE-TURBO + MULTIPLE LORA
# ============================================================

print("\n" + "=" * 60)
print("        Z-IMAGE-TURBO + MULTIPLE LoRA")
print("=" * 60)


# ============================================================
# COMFYUI NODES
# ============================================================

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()


# ============================================================
# BASE MODEL
# ============================================================

startup_start = time.time()

with torch.inference_mode():

    print("\n[1/3] Loading UNet... ", end="", flush=True)

    t0 = time.time()

    base_model = UNETLoader.load_unet(
        "z_image_turbo_bf16.safetensors",
        "default"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")


    print("[2/3] Loading CLIP (Qwen3)... ", end="", flush=True)

    t0 = time.time()

    base_clip = CLIPLoader.load_clip(
        "qwen_3_4b.safetensors",
        type="lumina2"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")


    print("[3/3] Loading VAE... ", end="", flush=True)

    t0 = time.time()

    vae = VAELoader.load_vae(
        "ae.safetensors"
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")


print(
    f"\n✅ Base models loaded in "
    f"{time.time() - startup_start:.1f}s"
)

print("=" * 60)


# ============================================================
# LORA DIRECTORY
# ============================================================

# Normal ComfyUI location
LORA_DIR = "./models/loras"


def get_lora_files():

    if not os.path.exists(LORA_DIR):
        print(
            f"\n⚠️ LoRA directory not found:"
            f"\n{os.path.abspath(LORA_DIR)}"
        )

        return [""]


    files = []

    for root, dirs, filenames in os.walk(LORA_DIR):

        for filename in filenames:

            if filename.lower().endswith(
                (".safetensors", ".pt", ".ckpt")
            ):

                # Return path relative to models/loras
                relative_path = os.path.relpath(
                    os.path.join(root, filename),
                    LORA_DIR
                )

                files.append(relative_path)


    files.sort()

    if not files:
        print(
            "\n⚠️ No LoRAs found in:"
            f"\n{os.path.abspath(LORA_DIR)}"
        )

        return [""]

    print(
        f"\n✅ Found {len(files)} LoRA(s)"
    )

    for f in files:
        print(f"   • {f}")

    return files


LORA_FILES = get_lora_files()


# ============================================================
# APPLY MULTIPLE LORAS
# ============================================================

def apply_loras(
    lora_names,
    lora_strengths,
    clip_strengths
):

    model = base_model
    clip = base_clip

    applied = []

    for i in range(len(lora_names)):

        lora_name = lora_names[i]

        if not lora_name:
            continue

        model_strength = float(
            lora_strengths[i]
        )

        clip_strength = float(
            clip_strengths[i]
        )


        # Skip completely disabled LoRA
        if model_strength == 0 and clip_strength == 0:
            continue


        print(
            f"\n   [{i + 1}] Applying LoRA:"
        )

        print(
            f"       {lora_name}"
        )

        print(
            f"       Model strength: "
            f"{model_strength}"
        )

        print(
            f"       CLIP strength: "
            f"{clip_strength}"
        )


        t0 = time.time()


        model, clip = LoraLoader.load_lora(
            model,
            clip,
            lora_name,
            model_strength,
            clip_strength
        )


        print(
            f"       done "
            f"({time.time() - t0:.1f}s)"
        )


        applied.append(
            lora_name
        )


    return model, clip, applied


# ============================================================
# SAVE HELPERS
# ============================================================

save_dir = "./results"

os.makedirs(
    save_dir,
    exist_ok=True
)


def get_save_path(prompt):

    safe_prompt = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        prompt
    )[:25]

    uid = uuid.uuid4().hex[:6]

    filename = (
        f"{safe_prompt}_{uid}.png"
    )

    return os.path.join(
        save_dir,
        filename
    )


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate(input):

    values = input["input"]


    # --------------------------------------------------------
    # Basic settings
    # --------------------------------------------------------

    positive_prompt = values[
        "positive_prompt"
    ]

    negative_prompt = values[
        "negative_prompt"
    ]

    seed = values["seed"]

    steps = values["steps"]

    cfg = values["cfg"]

    sampler_name = values[
        "sampler_name"
    ]

    scheduler = values[
        "scheduler"
    ]

    denoise = values["denoise"]

    width = values["width"]

    height = values["height"]

    batch_size = values[
        "batch_size"
    ]


    # --------------------------------------------------------
    # LoRA settings
    # --------------------------------------------------------

    lora_names = values[
        "lora_names"
    ]

    lora_strengths = values[
        "lora_strengths"
    ]

    clip_strengths = values[
        "clip_strengths"
    ]


    print("\n" + "=" * 60)
    print("              NEW GENERATION")
    print("=" * 60)


    total_start = time.time()


    # ========================================================
    # APPLY LORAS
    # ========================================================

    print(
        "\n[1/5] Applying LoRAs..."
    )


    t0 = time.time()


    model, clip, applied_loras = apply_loras(
        lora_names,
        lora_strengths,
        clip_strengths
    )


    print(
        f"\n✅ Applied {len(applied_loras)} LoRA(s)"
    )

    print(
        f"   Time: {time.time() - t0:.1f}s"
    )


    # ========================================================
    # PROMPTS
    # ========================================================

    print(
        "\n[2/5] Encoding prompts... ",
        end="",
        flush=True
    )


    t0 = time.time()


    positive = CLIPTextEncode.encode(
        clip,
        positive_prompt
    )[0]


    negative = CLIPTextEncode.encode(
        clip,
        negative_prompt
    )[0]


    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # LATENT
    # ========================================================

    print(
        "[3/5] Creating latent image... ",
        end="",
        flush=True
    )


    t0 = time.time()


    latent_image = EmptyLatentImage.generate(
        int(width),
        int(height),
        batch_size=int(batch_size)
    )[0]


    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # SAMPLING
    # ========================================================

    print(
        f"[4/5] Sampling "
        f"({steps} steps)..."
    )


    t0 = time.time()


    samples = KSampler.sample(
        model,
        int(seed),
        int(steps),
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=float(denoise)
    )[0]


    print(
        f"      Sampling done "
        f"({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # VAE DECODE
    # ========================================================

    print(
        "[5/5] Decoding image... ",
        end="",
        flush=True
    )


    t0 = time.time()


    decoded = VAEDecode.decode(
        vae,
        samples
    )[0].detach()


    print(
        f"done ({time.time() - t0:.1f}s)"
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_path = get_save_path(
        positive_prompt
    )


    image_array = np.array(
        decoded[0] * 255,
        dtype=np.uint8
    )


    Image.fromarray(
        image_array
    ).save(
        save_path
    )


    print(
        f"\n💾 Saved:"
        f"\n   {save_path}"
    )


    # ========================================================
    # GOOGLE DRIVE
    # ========================================================

    drive_path = (
        "/content/gdrive/MyDrive/"
        "z_image_turbo"
    )


    if os.path.exists(drive_path):

        shutil.copy(
            save_path,
            drive_path
        )

        print(
            f"☁️ Copied to Google Drive:"
            f"\n   {drive_path}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"\n🎨 LoRAs used:"
    )

    if applied_loras:

        for lora in applied_loras:
            print(f"   • {lora}")

    else:

        print("   • None")


    print(
        f"\n🌱 Seed: {seed}"
    )

    print(
        f"⏱️ Total: "
        f"{time.time() - total_start:.1f}s"
    )

    print("=" * 60 + "\n")


    return save_path, seed


# ============================================================
# GRADIO
# ============================================================

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

    lora1,
    lora1_strength,
    lora1_clip,

    lora2,
    lora2_strength,
    lora2_clip,

    lora3,
    lora3_strength,
    lora3_clip,

    lora4,
    lora4_strength,
    lora4_clip,

    lora5,
    lora5_strength,
    lora5_clip,

    sampler_name="euler",
    scheduler="simple"
):


    lora_names = [
        lora1,
        lora2,
        lora3,
        lora4,
        lora5
    ]


    lora_strengths = [
        lora1_strength,
        lora2_strength,
        lora3_strength,
        lora4_strength,
        lora5_strength
    ]


    clip_strengths = [
        lora1_clip,
        lora2_clip,
        lora3_clip,
        lora4_clip,
        lora5_clip
    ]


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

            "lora_names":
                lora_names,

            "lora_strengths":
                lora_strengths,

            "clip_strengths":
                clip_strengths
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


# ============================================================
# DEFAULT PROMPTS
# ============================================================

DEFAULT_POSITIVE = """
A beautiful woman with platinum blond hair that is almost white,
snowy white skin, red bush, very big plump red lips, high cheek
bones and sharp. She has almond shaped red eyes and she's holding
an intricate mask. She's wearing white and gold royal gown with
a black cloak. In the veins of her neck its gold.
"""


DEFAULT_NEGATIVE = """
low quality, blurry, unnatural skin tone, bad lighting,
pixelated, noise, oversharpen, soft focus
"""


# ============================================================
# CSS
# ============================================================

custom_css = """
.gradio-container {
    font-family:
        'SF Pro Display',
        -apple-system,
        BlinkMacSystemFont,
        sans-serif;
}

.lora-box {
    border: 1px solid #888;
    border-radius: 10px;
    padding: 10px;
}
"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:


    gr.HTML("""
<div style="
    width:100%;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    margin:20px 0;
">

<h1 style="font-size:2.5em; margin-bottom:10px;">
Z-Image-Turbo + Multiple LoRA
</h1>

</div>
""")


    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    with gr.Row():


        # ====================================================
        # LEFT
        # ====================================================

        with gr.Column():


            positive = gr.Textbox(
                DEFAULT_POSITIVE,
                label="Positive Prompt",
                lines=6
            )


            negative = gr.Textbox(
                DEFAULT_NEGATIVE,
                label="Negative Prompt",
                lines=5
            )


            # =================================================
            # IMAGE SETTINGS
            # =================================================

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


            with gr.Row():

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


            # =================================================
            # LORA SETTINGS
            # =================================================

            with gr.Accordion(
                "🎨 LoRA Settings",
                open=True
            ):


                gr.Markdown(
                    """
                    ### Stack multiple LoRAs

                    Leave a slot empty if you don't want to use it.
                    """
                )


                # ---------------------------------------------
                # LORA 1
                # ---------------------------------------------

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown("### LoRA 1")

                    lora1 = gr.Dropdown(
                        choices=LORA_FILES,
                        value=LORA_FILES[0],
                        label="LoRA"
                    )

                    with gr.Row():

                        lora1_strength = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="Model Strength"
                        )

                        lora1_clip = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="CLIP Strength"
                        )


                # ---------------------------------------------
                # LORA 2
                # ---------------------------------------------

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown("### LoRA 2")

                    lora2 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA"
                    )

                    with gr.Row():

                        lora2_strength = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="Model Strength"
                        )

                        lora2_clip = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="CLIP Strength"
                        )


                # ---------------------------------------------
                # LORA 3
                # ---------------------------------------------

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown("### LoRA 3")

                    lora3 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA"
                    )

                    with gr.Row():

                        lora3_strength = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="Model Strength"
                        )

                        lora3_clip = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="CLIP Strength"
                        )


                # ---------------------------------------------
                # LORA 4
                # ---------------------------------------------

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown("### LoRA 4")

                    lora4 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA"
                    )

                    with gr.Row():

                        lora4_strength = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="Model Strength"
                        )

                        lora4_clip = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="CLIP Strength"
                        )


                # ---------------------------------------------
                # LORA 5
                # ---------------------------------------------

                with gr.Group(
                    elem_classes="lora-box"
                ):

                    gr.Markdown("### LoRA 5")

                    lora5 = gr.Dropdown(
                        choices=[""] + LORA_FILES,
                        value="",
                        label="LoRA"
                    )

                    with gr.Row():

                        lora5_strength = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="Model Strength"
                        )

                        lora5_clip = gr.Slider(
                            -2.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="CLIP Strength"
                        )


            # =================================================
            # ADVANCED
            # =================================================

            with gr.Accordion(
                "⚙️ Advanced Settings",
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


            # =================================================
            # GENERATE
            # =================================================

            run = gr.Button(
                "🚀 Generate",
                variant="primary",
                size="lg"
            )


        # ====================================================
        # RIGHT
        # ====================================================

        with gr.Column():

            output_img = gr.Image(
                label="Generated Image",
                height=600
            )


            download_image = gr.File(
                label="Download Image"
            )


            used_seed = gr.Textbox(
                label="Seed Used",
                interactive=False
            )


    # ========================================================
    # BUTTON
    # ========================================================

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

            lora1,
            lora1_strength,
            lora1_clip,

            lora2,
            lora2_strength,
            lora2_clip,

            lora3,
            lora3_strength,
            lora3_clip,

            lora4,
            lora4_strength,
            lora4_clip,

            lora5,
            lora5_strength,
            lora5_clip
        ],

        outputs=[

            output_img,
            download_image,
            used_seed
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(
    share=True,
    debug=True
)
