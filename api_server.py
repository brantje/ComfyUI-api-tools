import logging
import os
import traceback
from urllib.parse import urlparse
import mimetypes
from aiohttp.web import FileResponse, Request, json_response
from server import PromptServer
import folder_paths

from .model_utils.refresh import refresh_folder
from .metrics.prometheus import get_metrics
from .model_utils.install import install_model_url

routes = PromptServer.instance.routes


def _normalize_rel_model_path(p: str) -> str:
    """
    Normalize a user-provided relative model path for comparison.
    - Accept both '/' and '\\' as separators (Windows clients often send '\\').
    - Reject absolute paths, traversal, and suspicious segments.
    """
    if not isinstance(p, str):
        raise ValueError("invalid model path")

    if p.startswith(("/", "\\")):
        raise ValueError("invalid model path")

    # Normalize separators to posix for consistent matching.
    p2 = p.replace("\\", "/")

    # Reject drive letters / URI-like / ADS-like patterns.
    if ":" in p2:
        raise ValueError("invalid model path")

    parts = p2.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("invalid model path")

    return "/".join(parts)


def success_resp(**kwargs):
    return json_response({"code": 200, "message": "success", **kwargs})


def error_resp(code, message, **kwargs):
    return json_response({"code": code, "message": message, **kwargs})

@routes.get("/api-tools/v1/models")
async def get_model_folders(request: Request):
    """ List all model folders """
    folders = list(folder_paths.folder_names_and_paths.keys())

    return success_resp(result={"folders": folders })

@routes.post("/api-tools/v1/models/install")
async def install_model(request: Request):
    """Install a model from url, filtered with a allow list  """
    json_data = await request.json()
    url = json_data.get('url')
    allowed_domains = ["civitai.com", "github.com", "raw.githubusercontent.com", "huggingface.co"]
    if url:
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if domain.startswith('www.'):
                domain = domain[4:]

        # Check if the domain is in the allowed list
            if domain not in allowed_domains:
                return error_resp(403, f"Domain '{domain}' is not in the allowed domains list")
        except Exception as e:
             return error_resp(400, f"Invalid URL format: {str(e)}")
    else:
        return error_resp(400, "URL is required")

    success, message = install_model_url(json_data)

    if not success:
        return error_resp(409, message)

    return success_resp()



@routes.get("/api-tools/v1/models/{folder}")
async def get_model_folder(request: Request):
    """ List all models in a folder """
    folder = request.match_info['folder']
    try:
        checkpoints = folder_paths.get_filename_list(folder)
        result = [
            {
                "filename": os.path.basename(ckpt),
                "path": ckpt,
                "full_path": folder_paths.get_full_path(folder, ckpt) or ckpt,
            }
            for ckpt in checkpoints
        ]
        return success_resp(result=result)
    except Exception as e:
        return error_resp(500, str('Path not found'))


@routes.post("/api-tools/v1/models/{folder}/refresh")
async def refresh_checkpoints(request: Request):
    """Refresh the checkpoints list and return the updated list(maybe useful for models that stored in a network storage)"""
    folder = request.match_info['folder']
    try:
        data = refresh_folder(folder)
        return success_resp(data=data)
    except Exception as e:
        return error_resp(500, str(e))


@routes.delete("/api-tools/v1/models/{folder}/{model:.+}")
async def remove_model(request: Request):
    """Removes a model from the file system """
    input_folder = request.match_info['folder']
    model_input = request.match_info['model']
    try:
        # Basic traversal checks
        try:
            normalized_model_input = _normalize_rel_model_path(model_input)
        except ValueError as e:
            return error_resp(400, str(e))

        # Get the base folder path for this model type
        folder_paths_list = folder_paths.folder_names_and_paths.get(input_folder)
        if not folder_paths_list:
            return error_resp(404, f"Folder '{input_folder}' not found")

        base_folder_path = folder_paths_list[0][0]  # First path in the list

        # Walk through the folder tree to find the matching file
        matched_full_path = None

        for root, dirs, files in os.walk(base_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Get the relative path from the base folder
                rel_path = os.path.relpath(file_path, base_folder_path)
                # Normalize separators for comparison
                rel_path_normalized = rel_path.replace("\\", "/")

                # Check if this file matches the requested model path
                if (rel_path == model_input or
                    rel_path == normalized_model_input or
                    rel_path_normalized == normalized_model_input):
                    matched_full_path = file_path
                    break
            if matched_full_path:
                break

        if matched_full_path is None:
            return error_resp(404, 'Model not found')

        # Additional security check: ensure the matched path is still within the base folder
        if not os.path.commonpath([os.path.abspath(matched_full_path), os.path.abspath(base_folder_path)]) == os.path.abspath(base_folder_path):
            return error_resp(403, "Access denied")

        if not os.path.isfile(matched_full_path):
            return error_resp(404, 'Not Found')

        os.remove(matched_full_path)
        return success_resp(result=matched_full_path)

    except Exception as e:
        return error_resp(500, str(e))


@routes.get("/api-tools/v1/models/{folder}/{model:.+}/download")
async def download_model(request: Request):
    """Download a model file from the file system."""
    input_folder = request.match_info.get("folder")
    model_input = request.match_info.get("model")

    if not input_folder:
        return error_resp(400, "folder is required")

    if not model_input:
        return error_resp(400, "model is required")

    # Basic traversal checks; final enforcement is done by validating the path is present in ComfyUI's model index.
    try:
        normalized_model_input = _normalize_rel_model_path(model_input)
    except ValueError as e:
        return error_resp(400, str(e))

    try:
        folder_paths_list = folder_paths.folder_names_and_paths.get(input_folder)
        if not folder_paths_list:
            return error_resp(404, f"Folder '{input_folder}' not found")

        base_folder_path = folder_paths_list[0][0]  # First path in the list

        # Walk through the folder tree to find the matching file
        # This handles subdirectories properly for both Windows and Linux
        matched_full_path = None

        for root, dirs, files in os.walk(base_folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Get the relative path from the base folder
                rel_path = os.path.relpath(file_path, base_folder_path)
                # Normalize separators for comparison
                rel_path_normalized = rel_path.replace("\\", "/")

                # Check if this file matches the requested model path
                if (rel_path == model_input or
                    rel_path == normalized_model_input or
                    rel_path_normalized == normalized_model_input):
                    matched_full_path = file_path
                    break
            if matched_full_path:
                break

        if matched_full_path is None:
            return error_resp(404, "Model not found")

        # Additional security check: ensure the matched path is still within the base folder
        if not os.path.commonpath([os.path.abspath(matched_full_path), os.path.abspath(base_folder_path)]) == os.path.abspath(base_folder_path):
            return error_resp(403, "Access denied")

        if not os.path.isfile(matched_full_path):
            return error_resp(404, "Not Found")

        content_type, _ = mimetypes.guess_type(matched_full_path)
        resp = FileResponse(path=matched_full_path)
        resp.headers["Content-Disposition"] = f'attachment; filename="{os.path.basename(matched_full_path)}"'
        if content_type:
            resp.content_type = content_type
        return resp
    except Exception as e:
        logging.error("Error downloading model", exc_info=True)
        return error_resp(500, str(e))


@routes.get("/api-tools/v1/images/output")
async def get_output_images(request: Request):
    try:
        is_temp = request.rel_url.query.get("temp", "false") == "true"
        folder = (
            folder_paths.get_temp_directory()
            if is_temp
            else folder_paths.get_output_directory()
        )
        # iterate through the folder and get the list of images
        images = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    image = {"name": file, "full_path": os.path.join(root, file)}
                    images.append(image)
        return success_resp(images=images)
    except Exception as e:
        return error_resp(500, str(e))



@routes.delete("/api-tools/v1/images/output/{filename}")
async def delete_output_images(request: Request):
    try:
        filename = request.match_info.get("filename")
        if filename is None:
            return error_resp(400, "filename is required")

        if filename[0] == "/" or ".." in filename:
            return error_resp(400, "invalid filename")

        is_temp = request.rel_url.query.get("temp", "false") == "true"
        annotated_file = f"{filename} [{'temp' if is_temp else 'output'}]"
        if not folder_paths.exists_annotated_filepath(annotated_file):
            return error_resp(404, f"file {filename} not found")

        filepath = folder_paths.get_annotated_filepath(annotated_file)
        os.remove(filepath)
        return success_resp()
    except Exception as e:
        return error_resp(500, str(e))


@routes.get("/api-tools/v1/images/input")
async def get_input_images(request: Request):
    try:
        folder = folder_paths.get_input_directory()

        # iterate through the folder and get the list of images
        images = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    image = {"name": file, "full_path": os.path.join(root, file)}
                    images.append(image)
        return success_resp(images=images)
    except Exception as e:
        return error_resp(500, str(e))


@routes.delete("/api-tools/v1/images/input/{filename}")
async def delete_input_images(request: Request):
    try:
        filename = request.match_info.get("filename")
        if filename is None:
            return error_resp(400, "filename is required")

        if filename[0] == '/' or '..' in filename:
            return error_resp(400, "invalid filename")

        is_temp = request.rel_url.query.get("temp", "false") == "true"
        annotated_file = f"{filename} [{'temp' if is_temp else 'input'}]"
        if not folder_paths.exists_annotated_filepath(annotated_file):
            return error_resp(404, f"file {filename} not found")

        filepath = folder_paths.get_annotated_filepath(annotated_file)
        os.remove(filepath)
        return success_resp()
    except Exception as e:
        return error_resp(500, str(e))

@routes.get("/api-tools/v1/metrics")
async def get_prometheus_metrics(request: Request):
    """Return Prometheus metrics"""
    try:
        metrics_text = get_metrics()
        return json_response(text=metrics_text, content_type="text/plain")
    except Exception as e:
        logging.error(f"Error generating metrics: {str(e)}")
        return error_resp(500, f"Error generating metrics: {str(e)}")


def run_comfyui_api_tools():
    print("extra API server started")