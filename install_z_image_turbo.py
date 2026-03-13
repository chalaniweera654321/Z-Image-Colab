# root_path = "/content"  # @param ['/content', '/root', '/kaggle/working','/teamspace/studios/this_studio']
# %cd $root_path/shorts_podcast

import os
import sys
import subprocess
from hf_mirror import download_file


# --------------------------------------------------
# Helper runner
# --------------------------------------------------
def run(cmd, check=True):
    print("\n" + "=" * 70)
    print("RUN:", " ".join(cmd))
    print("CWD:", os.getcwd())
    print("=" * 70)

    subprocess.run(cmd, check=check)


def works(cmd):
    try:
        subprocess.run(cmd,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except:
        return False


# --------------------------------------------------
# Linux dependencies
# --------------------------------------------------
def linux_install():
    print("\n🔧 Installing system packages...")

    if works(["sudo", "-n", "true"]):
        installer = ["sudo", "apt"]
    elif works(["apt", "--version"]):
        installer = ["apt"]
    else:
        print("⚠️ apt not available")
        return

    packages = ["ffmpeg", "aria2"]

    run(installer + ["update"], check=False)
    run(installer + ["install", "-y"] + packages, check=False)


# --------------------------------------------------
# aria2 download + HF fallback
# --------------------------------------------------
def aria_download(url, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)

    cmd = [
        "aria2c",
        "--console-log-level=warn",
        "-c",
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        url,
        "-d", os.path.dirname(output),
        "-o", os.path.basename(output),
    ]

    print("\n⬇️ PRIMARY DOWNLOAD Python")
    print("URL :", url)
    print("SAVE:", output)

    try:
        download_file(url, output, redownload=True)
        print("✅ HF download success")
    except Exception as hf_error:
        print("HF Error:", hf_error)
        print("\n⚠️ hf_hub FAILED → switching to aria2")
        try:
            subprocess.run(cmd, check=True)
            print("✅ aria2 fallback download success")
        except Exception as e:
            print("Error:", e)
            print("\n❌ BOTH DOWNLOAD METHODS FAILED")
            raise RuntimeError("Download completely failed")


# --------------------------------------------------
# Model download
# --------------------------------------------------
def download_models(comfy_path):
    print("\n📦 Downloading models to:", comfy_path)

    files = [
        {
            "url": "https://huggingface.co/T5B/Z-Image-Turbo-FP8/resolve/main/z-image-turbo-fp8-e4m3fn.safetensors",
            "path": f"{comfy_path}/models/diffusion_models/z-image-turbo-fp8-e4m3fn.safetensors",
        },
        {
            "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
            "path": f"{comfy_path}/models/clip/qwen_3_4b.safetensors",
        },
        {
            "url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            "path": f"{comfy_path}/models/vae/ae.safetensors",
        },
    ]

    for f in files:
        aria_download(f["url"], f["path"])


# --------------------------------------------------
# Main installer
# --------------------------------------------------
def z_image():
    repo_root = os.getcwd()
    parent_dir = os.path.dirname(repo_root)

    comfy_path = os.path.join(parent_dir, "ComfyUI")

    print("\n🔥 INSTALL START")
    print("Repo folder :", repo_root)
    print("Parent dir  :", parent_dir)
    print("ComfyUI path:", comfy_path)

    # System deps
    linux_install()

    # Clone ComfyUI outside repo
    if not os.path.exists(comfy_path):
        run([
            "git", "clone",
            "https://github.com/comfyanonymous/ComfyUI",
            comfy_path,
        ])
    else:
        print("✅ ComfyUI already exists")

    # Pin stable commit
    os.chdir(comfy_path)
    run(["git", "fetch", "--all", "-q"])
    run([
        "git", "reset", "--hard",
        "3c8456223c5f6a41af7d99219b391c8c58acb552"
    ])

    # Download models
    download_models(comfy_path)

    # ComfyUI requirements
    os.chdir(comfy_path)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=False)

    run([sys.executable, "-m", "pip", "install", "gradio"],
        check=False)
    print("\n🎉 INSTALL COMPLETE")


z_image()
