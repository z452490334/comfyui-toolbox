"""
MuAPI GPT-Image-2 ComfyUI Nodes
================================
ComfyUI nodes for GPT-Image-2 image generation via muapi.ai.

  GPTImage2TextToImage     — POST /api/v1/gpt-image-2-text-to-image
  GPTImage2ImageToImage    — POST /api/v1/gpt-image-2-image-to-image

Auth:     x-api-key header
Polling:  GET /api/v1/predictions/{request_id}/result
Upload:   POST /api/v1/upload_file
"""

import io
import os
import time

import numpy as np
import requests
import torch
from PIL import Image

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_CONFIG_PATH = os.path.join(PLUGIN_DIR, "config.json")
LEGACY_CONFIG_PATH = os.path.expanduser("~/.muapi/config.json")
DEFAULT_BASE_URL = "https://api.muapi.ai/api/v1"
POLL_INTERVAL = 5
MAX_WAIT = 600


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_config_value(key):
    for config_path in (PLUGIN_CONFIG_PATH, LEGACY_CONFIG_PATH):
        if os.path.isfile(config_path):
            try:
                import json as _json
                with open(config_path, encoding="utf-8") as f:
                    value = _json.load(f).get(key, "")
                if value:
                    return str(value).strip()
            except Exception:
                pass
    return ""


def _load_api_key(api_key_input):
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()
    key = _load_config_value("api_key")
    if key:
        return key
    raise RuntimeError(
        "No API key found. Either paste your key into the api_key field, "
        f"create {PLUGIN_CONFIG_PATH}, or run "
        "`muapi auth configure --api-key YOUR_KEY` in a terminal."
    )


def _load_base_url(base_url_input=""):
    if base_url_input and base_url_input.strip():
        return base_url_input.strip().rstrip("/")

    for env_name in ("GPT_IMAGE2_BASE_URL", "MUAPI_BASE_URL"):
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url.rstrip("/")

    base_url = _load_config_value("base_url")
    if base_url:
        return base_url.rstrip("/")

    return DEFAULT_BASE_URL


def _upload_image(api_key, image_tensor, base_url):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)
    resp = requests.post(
        f"{base_url}/upload_file",
        headers={"x-api-key": api_key},
        files={"file": ("image.jpg", buf, "image/jpeg")},
        timeout=120,
    )
    _check(resp)
    return _url(resp.json())


def _url(data):
    u = data.get("url") or data.get("file_url") or data.get("output")
    if not u:
        raise RuntimeError(f"Upload missing URL: {data}")
    return str(u)


def _submit(api_key, base_url, endpoint, payload):
    resp = requests.post(
        f"{base_url}/{endpoint}",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    _check(resp)
    rid = resp.json().get("request_id")
    if not rid:
        raise RuntimeError(f"No request_id in response: {resp.json()}")
    return rid


def _poll(api_key, base_url, request_id):
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        resp = requests.get(
            f"{base_url}/predictions/{request_id}/result",
            headers={"x-api-key": api_key},
            timeout=30,
        )
        _check(resp)
        data = resp.json()
        status = data.get("status")
        print(f"[GPTImage2] {status}  {request_id}")
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"Generation failed: {data.get('error', 'unknown')}")
        time.sleep(POLL_INTERVAL)
    raise RuntimeError(f"Timeout waiting for result: {request_id}")


def _output_image_url(result):
    out = result.get("outputs") or result.get("output") or []
    if isinstance(out, list) and out:
        return str(out[0])
    if isinstance(out, str):
        return out
    for k in ("image_url", "url"):
        if result.get(k):
            return str(result[k])
    raise RuntimeError(f"No output image URL in result: {result}")


def _download_image(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _check(resp):
    if resp.status_code == 401:
        raise RuntimeError("Auth failed — check your API key.")
    if resp.status_code == 402:
        raise RuntimeError("Insufficient credits — top up at muapi.ai.")
    if resp.status_code == 429:
        raise RuntimeError("Rate limited — please retry later.")
    if not resp.ok:
        print(f"[GPTImage2] API ERROR {resp.status_code}: {resp.text[:500]}")
        try:
            err = resp.json()
            raise RuntimeError(f"API {resp.status_code}: {err}")
        except Exception:
            raise RuntimeError(f"API {resp.status_code}: {resp.text[:300]}")


# ── Nodes ──────────────────────────────────────────────────────────────────────

class GPTImage2ApiKey:
    """
    Store your MuAPI API key once and wire it to any GPT-Image-2 node.
    Alternatively, leave all api_key fields empty — nodes auto-read from
    this plugin's config.json, then ~/.muapi/config.json.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Your muapi.ai API key. Get one at muapi.ai → Dashboard → API Keys",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "run"
    CATEGORY = "🖼️ GPT-Image-2"

    def run(self, api_key):
        return (_load_api_key(api_key),)


class GPTImage2BaseUrl:
    """
    Provide a MuAPI-compatible API base URL.
    Defaults to https://api.muapi.ai/api/v1 when left blank.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_BASE_URL,
                        "tooltip": "MuAPI-compatible API base URL, for example https://api.example.com/api/v1",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base_url",)
    FUNCTION = "run"
    CATEGORY = "🖼️ GPT-Image-2"

    def run(self, base_url):
        return (_load_base_url(base_url),)


class GPTImage2TextToImage:
    """
    GPT-Image-2 Text-to-Image
    --------------------------
    Generate a high-quality image from a text prompt using GPT-Image-2.

    Endpoint: POST /api/v1/gpt-image-2-text-to-image
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A photorealistic image of a red fox sitting in a snowy forest at dusk.",
                    },
                ),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = "🖼️ GPT-Image-2"

    def run(self, prompt, api_key="", base_url=""):
        api_key = _load_api_key(api_key)
        base_url = _load_base_url(base_url)
        payload = {"prompt": prompt}
        print(f"[GPTImage2 T2I] Submitting to {base_url}...")
        rid = _submit(api_key, base_url, "gpt-image-2-text-to-image", payload)
        result = _poll(api_key, base_url, rid)
        url = _output_image_url(result)
        print(f"[GPTImage2 T2I] Done → {url}")
        image = _download_image(url)
        return (image, url, rid)


class GPTImage2ImageToImage:
    """
    GPT-Image-2 Image-to-Image
    ---------------------------
    Transform or edit up to 9 reference images guided by a text prompt.
    Common uses: style transfer, product shots, scene editing.

    Endpoint: POST /api/v1/gpt-image-2-image-to-image

    Example prompt:
        "Transform this product image into a premium e-commerce poster style."
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Transform this product image into a premium e-commerce poster style.",
                    },
                ),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = "🖼️ GPT-Image-2"

    def run(
        self,
        prompt,
        api_key="",
        base_url="",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
    ):
        api_key = _load_api_key(api_key)
        base_url = _load_base_url(base_url)
        tensors = [image_1, image_2, image_3, image_4, image_5,
                   image_6, image_7, image_8, image_9]
        images_list = []
        for i, img in enumerate(tensors, 1):
            if img is not None:
                print(f"[GPTImage2 I2I] Uploading image {i}...")
                images_list.append(_upload_image(api_key, img, base_url))
        if not images_list:
            raise ValueError("At least one input image is required.")

        payload = {"prompt": prompt, "images_list": images_list}
        print(f"[GPTImage2 I2I] Submitting ({len(images_list)} image(s)) to {base_url}...")
        rid = _submit(api_key, base_url, "gpt-image-2-image-to-image", payload)
        result = _poll(api_key, base_url, rid)
        url = _output_image_url(result)
        print(f"[GPTImage2 I2I] Done → {url}")
        image = _download_image(url)
        return (image, url, rid)


NODE_CLASS_MAPPINGS = {
    "GPTImage2ApiKey":        GPTImage2ApiKey,
    "GPTImage2BaseUrl":       GPTImage2BaseUrl,
    "GPTImage2TextToImage":   GPTImage2TextToImage,
    "GPTImage2ImageToImage":  GPTImage2ImageToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2ApiKey":        "🔑 GPT-Image-2 API Key",
    "GPTImage2BaseUrl":       "🌐 GPT-Image-2 Base URL",
    "GPTImage2TextToImage":   "🖼️ GPT-Image-2 Text to Image",
    "GPTImage2ImageToImage":  "🖼️ GPT-Image-2 Image to Image",
}
