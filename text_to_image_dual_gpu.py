import sys, os, random, time, torch, uuid, re, gc
import numpy as np
from PIL import Image
import re, uuid

def image_file_name(prompt):
    # Remove punctuation, keep letters & numbers
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)

    # Lowercase + replace spaces with _
    clean = clean.lower().strip().replace(" ", "_")

    # Limit length (optional but good)
    clean = clean[:30]

    # Add random suffix
    uid = uuid.uuid4().hex[:8]

    return f"{clean}_{uid}.png"
# =================================================
# 1. SETUP COMFYUI PATHS
# =================================================
current_dir = os.getcwd()
root_path = os.path.dirname(current_dir)
base_path="."

ComfyUI = f"{root_path}/ComfyUI"
sys.path.append(ComfyUI)

from nodes import NODE_CLASS_MAPPINGS
import comfy.model_management

# Initialize Loader Classes (Lightweight, no VRAM used yet)
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# =================================================
# 2. GPU DETECTION LOGIC
# =================================================
def gpu_vram_gb():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

def is_low_gpu():
    # Returns True if VRAM is <= 15GB
    return gpu_vram_gb() <= 15

def has_two_gpus():
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2

# Global Flags
LOW_GPU = is_low_gpu()
DUAL_GPU = has_two_gpus()
_Z_IMAGE_MODELS = None

# =================================================
# 3. MODEL MANAGEMENT (LOAD / UNLOAD)
# =================================================
def load_z_models():
    """
    Loads models into VRAM. 
    Smartly detects if Single or Dual GPU is available.
    """
    global _Z_IMAGE_MODELS
    
    if _Z_IMAGE_MODELS is not None:
        return _Z_IMAGE_MODELS

    print("--- Loading Z-Image Models ---")
    
    # GPU Assignment Logic
    gpu_count = torch.cuda.device_count()
    
    if gpu_count > 1:
        print(f"Dual GPU Detected ({gpu_count} GPUs). Splitting models.")
        device_unet = torch.device("cuda:1")
        device_clip_vae = torch.device("cuda:0")
    else:
        print(f"Single GPU Detected ({gpu_vram_gb():.1f} GB). Loading everything on cuda:0.")
        device_unet = torch.device("cuda:0")
        device_clip_vae = torch.device("cuda:0")

    with torch.inference_mode():
        print(f"Loading UNET to {device_unet}...")
        unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
        unet.load_device = device_unet
        unet.offload_device = torch.device("cpu") 
        
        print(f"Loading CLIP and VAE to {device_clip_vae}...")
        clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
        clip.load_device = device_clip_vae
        
        vae = VAELoader.load_vae("ae.safetensors")[0]
        vae.load_device = device_clip_vae
        
    _Z_IMAGE_MODELS = (unet, clip, vae)
    print("--- Models Loaded Successfully ---")
    return _Z_IMAGE_MODELS

def unload_z_models():
    """
    Deletes models, clears VRAM, and runs garbage collection.
    """
    global _Z_IMAGE_MODELS
    
    print("--- Unloading Z-Image Models ---")
    if _Z_IMAGE_MODELS is not None:
        del _Z_IMAGE_MODELS
        _Z_IMAGE_MODELS = None
        
    # Python Garbage Collection
    gc.collect()
    
    # Torch VRAM cleanup
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # ComfyUI specific cleanup
    comfy.model_management.soft_empty_cache()
    
    print("--- GPU Memory Cleared ---")

save_dir=f"{current_dir}/z_image_results"
os.makedirs(save_dir, exist_ok=True)

# =================================================
# 4. GENERATION LOGIC
# =================================================
@torch.inference_mode()
def generate(input_data, model_data):
    unet, clip, vae = model_data
    v = input_data["input"]
    
    cond = CLIPTextEncode.encode(clip, v['positive_prompt'])[0]
    uncond = CLIPTextEncode.encode(clip, v['negative_prompt'])[0]
    
    latent = EmptyLatentImage.generate(v['width'], v['height'], batch_size=v['batch_size'])[0]
    
    print(f"Sampling (Seed: {v['seed']})...")
    samples = KSampler.sample(unet, v['seed'], v['steps'], v['cfg'], v['sampler_name'], 
                              v['scheduler'], cond, uncond, latent, denoise=v['denoise'])[0]

    print("Decoding...")
    decoded = VAEDecode.decode(vae, samples)[0].detach()
    image_path=image_file_name(v['positive_prompt'])
    save_path = os.path.join(save_dir, image_path)
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(save_path)
    
    return save_path, v['seed']

def z_image_turbo(positive_prompt,negative_prompt="",width=1280, height=720, steps=9,cfg=1.0,seed=0,model_data=None):
    """
    Main Interface.
    Logic:
    1. If model_data provided -> Use it (Fast Mode).
    2. If model_data None:
       - If Low GPU & Single GPU -> Auto-Load, Generate, Auto-Unload (Safe Mode).
       - Else -> Auto-Load, Generate, Keep Loaded.
    """
    
    force_unload_after = False
    
    # --- SMART LOADING LOGIC ---
    if model_data is None:
        print("Warning: No model_data provided.")
        
        if LOW_GPU and not DUAL_GPU:
            print("Action: Low GPU Detected. Auto-loading for single generation (Safe Mode).")
            model_data = load_z_models()
            force_unload_after = True
        else:
            print("Action: High GPU/Dual GPU. Loading models (will stay in memory).")
            model_data = load_z_models()
            force_unload_after = False
    # ---------------------------

    positive_prompt = positive_prompt
    if negative_prompt=="":
        negative_prompt = 'low resolution, blurry, out of focus, soft focus, pixelated, jpeg artifacts, compression artifacts, noise, grain, banding, aliasing, oversharpened, motion blur, ghosting, double exposure, smearing, bad anatomy, distorted anatomy, deformed body, warped proportions'
    if seed==0:
        random.randint(0, 1000000000)
    input_payload = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": 1,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
        }
    }

    try:
        image_path, seed = generate(input_payload, model_data)
    finally:
        # If we determined we need to clean up to save RAM (Low GPU mode)
        if force_unload_after:
            unload_z_models()

    return image_path
    
def z_image_set_up():
  # Condition: If we have LOW RAM (<=15GB) AND only 1 GPU, use Safe Mode.
  use_safe_mode = is_low_gpu() and not has_two_gpus()

  my_models = None

  if not use_safe_mode:
      # FAST MODE (Dual GPU or High VRAM)
      # We load models ONCE here. They stay in RAM for the whole loop.
      print("🚀 High Performance Detected: Loading models once for batch...")
      my_models = load_z_models()
  else:
      # SAFE MODE (Single Low GPU)
      # We keep my_models as None.
      # The 'z_image_turbo' function will see 'None', load the model, generate, 
      # and then UNLOAD it immediately for every single image.
      print("🛡️ Safe Mode Detected (Low VRAM): Will load/unload per image.")
      my_models = None
  return my_models

# # main.py
# import time
# from image_generation import (
#     z_image_turbo, 
#     load_z_models, 
#     unload_z_models, 
#     is_low_gpu, 
#     has_two_gpus
# )

# # =================================================
# # 1. SETUP BATCH STRATEGY
# # =================================================

# # Condition: If we have LOW RAM (<=15GB) AND only 1 GPU, use Safe Mode.
# use_safe_mode = is_low_gpu() and not has_two_gpus()

# my_models = None

# if not use_safe_mode:
#     # FAST MODE (Dual GPU or High VRAM)
#     # We load models ONCE here. They stay in RAM for the whole loop.
#     print("🚀 High Performance Detected: Loading models once for batch...")
#     my_models = load_z_models()
# else:
#     # SAFE MODE (Single Low GPU)
#     # We keep my_models as None.
#     # The 'z_image_turbo' function will see 'None', load the model, generate, 
#     # and then UNLOAD it immediately for every single image.
#     print("🛡️ Safe Mode Detected (Low VRAM): Will load/unload per image.")
#     my_models = None

# # =================================================
# # 2. RUN BATCH
# # =================================================

# prompts = [
#     "a cyberpunk city in rain", 
#     "a portrait of a wizard", 
#     "a futuristic race car"
# ]

# for p in prompts:
#     print(f"\n--- Generating: {p} ---")
    
#     # Pass 'my_models'. 
#     # If it is data -> Fast generation.
#     # If it is None -> Safe generation (Auto Load/Unload).
#     img_path = z_image_turbo(p, model_data=my_models)
    
#     print(f"Saved: {img_path}")

# # =================================================
# # 3. FINAL CLEANUP
# # =================================================
# print("\nBatch finished. Performing final cleanup...")
# # This ensures memory is freed regardless of which mode was used
# unload_z_models()





import gradio as gr
# from image_generation import (
#     z_image_turbo, 
#     load_z_models, 
#     unload_z_models, 
#     is_low_gpu, 
#     has_two_gpus
# )


# Condition: If we have LOW RAM (<=15GB) AND only 1 GPU, use Safe Mode.
use_safe_mode = is_low_gpu() and not has_two_gpus()

MODEL_DATA = None
if not use_safe_mode:
    # FAST MODE (Dual GPU or High VRAM)
    # We load models ONCE here. They stay in RAM for the whole loop.
    print("🚀 High Performance Detected: Loading models once for batch...")
    MODEL_DATA = load_z_models()
else:
    # SAFE MODE (Single Low GPU)
    # We keep my_models as None.
    # The 'z_image_turbo' function will see 'None', load the model, generate, 
    # and then UNLOAD it immediately for every single image.
    print("🛡️ Safe Mode Detected (Low VRAM): Will load/unload per image.")
    MODEL_DATA = None


def generate_image(
    positive_prompt,
    negative_prompt,
    width,
    height,
    steps,
    cfg,
    seed,
):
    img_path = z_image_turbo(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        model_data=MODEL_DATA   # <- keep models loaded
    )

    return img_path


custom_css = """.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"""



ASPECTS = {
    "1024x1024 (1:1)": (1024, 1024),
    "1152x896 (9:7)": (1152, 896),
    "896x1152 (7:9)": (896, 1152),
    "1152x864 (4:3)": (1152, 864),
    "864x1152 (3:4)": (864, 1152),
    "1248x832 (3:2)": (1248, 832),
    "832x1248 (2:3)": (832, 1248),
    "1280x720 (16:9)": (1280, 720),
    "720x1280 (9:16)": (720, 1280),
    "1344x576 (21:9)": (1344, 576),
    "576x1344 (9:21)": (576, 1344),
}

def update_dims(aspect):
    return ASPECTS[aspect]

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
          <div style="text-align: center; margin: 20px auto; max-width: 800px;">
              <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎨 Z Image Turbo Gradio API</h1>
          </div>""")
    with gr.Row():

        # LEFT SIDE INPUTS
        with gr.Column(scale=1):
            positive = gr.Textbox(
                label="Positive Prompt",
                placeholder="Describe your image...",
                lines=4   
            )
            aspect = gr.Dropdown(
                list(ASPECTS.keys()),
                value="1024x1024 (1:1)",
                label="Aspect Ratio"
            )
            

        
            generate_btn = gr.Button("🚀 Generate Image")
            with gr.Accordion("Settings", open=False):
              with gr.Row():
                width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                height = gr.Slider(256, 2048, value=1024, step=64, label="Height")

                # auto update width/height from dropdown
                aspect.change(
                    update_dims,
                    inputs=aspect,
                    outputs=[width, height]
                )
              with gr.Row():
                steps = gr.Slider(1, 30, value=9, label="Steps")
                cfg = gr.Slider(
                          0.1, 10,
                          value=1.0,
                          step=0.1,
                          label="CFG Scale"
                      )
                seed = gr.Number(
                          value=0,
                          label="Seed (0 = Random)"
                      )
              negative = gr.Textbox(label="Negative Prompt",lines=4)

        # RIGHT SIDE OUTPUT
        with gr.Column(scale=1):
            output_img = gr.Image(label="Generated Image",type="filepath")

    generate_btn.click(
        generate_image,
        inputs=[positive, negative, width, height, steps, cfg, seed],
        outputs=output_img
    )



# demo.launch(debug=True)
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def run_demo(share,debug):
    demo.queue().launch(share=share,debug=debug)
if __name__ == "__main__":
    run_demo()
