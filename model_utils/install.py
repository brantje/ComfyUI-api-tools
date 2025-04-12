import sys
import os
import urllib
import requests
import folder_paths


model_dir_name_map = {
    "checkpoints": "checkpoints",
    "checkpoint": "checkpoints",
    "unclip": "checkpoints",
    "text_encoders": "text_encoders",
    "clip": "text_encoders",
    "vae": "vae",
    "lora": "loras",
    "t2i-adapter": "controlnet",
    "t2i-style": "controlnet",
    "controlnet": "controlnet",
    "clip_vision": "clip_vision",
    "gligen": "gligen",
    "upscale": "upscale_models",
    "embedding": "embeddings",
    "embeddings": "embeddings",
    "unet": "diffusion_models",
    "diffusion_model": "diffusion_models",
}

def download_url_with_agent(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req)
        data = response.read()

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path, 'wb') as f:
            f.write(data)

    except Exception as e:
        print(f"Download error: {url} / {e}", file=sys.stderr)
        return False

    print("Installation was successful.")
    return True

def get_install_dir(data):
    if 'download_model_base' in folder_paths.folder_names_and_paths:
        models_base = folder_paths.folder_names_and_paths['download_model_base'][0][0]
    else:
        models_base = folder_paths.models_dir

    if data.get('save_path') != 'default':
        base_model = os.path.join(models_base, data['save_path'])
    else:
        model_dir_name = model_dir_name_map.get(data['type'].lower())
        if model_dir_name is not None:
            base_model = folder_paths.folder_names_and_paths[model_dir_name][0][0]
        else:
            base_model = os.path.join(models_base, "etc")

    return base_model

def install_model_url(json_data):
    install_dir = get_install_dir(json_data)
    save_path = os.path.join(install_dir, json_data.get('filename'))
    if os.path.exists(save_path):
        return False, f"File already exists at path: {save_path}"

    download_url_with_agent(json_data.get('url'), save_path)
    return True, "Model installed successfully"
