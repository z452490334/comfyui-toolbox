"""
OpenAI GPT Image ComfyUI Nodes
==============================
ComfyUI nodes for GPT Image generation via the OpenAI Images API.

  GPTImage2TextToImage     — POST /v1/images/generations
  GPTImage2ImageToImage    — POST /v1/images/edits

Auth:     Authorization: Bearer <api key>
"""

import base64
import io
import os
import json

import numpy as np
import requests
import torch
from PIL import Image

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_CONFIG_PATH = os.path.join(PLUGIN_DIR, "config.json")
LEGACY_CONFIG_PATH = os.path.expanduser("~/.muapi/config.json")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JIMENG_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_USER_AGENT = "gpt-image-2-comfyui/1.0"
TOOLBOX_CATEGORY = "⭐ Toolbox"
GPT_IMAGE_CATEGORY = f"{TOOLBOX_CATEGORY}/GPT-Image-2"
JIMENG_CATEGORY = f"{TOOLBOX_CATEGORY}/Jimeng"

MODEL_OPTIONS = ["gpt-image-2", "gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"]
SIZE_OPTIONS = [
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
]
QUALITY_OPTIONS = ["auto", "low", "medium", "high"]
BACKGROUND_OPTIONS = ["auto", "transparent", "opaque"]
OUTPUT_FORMAT_OPTIONS = ["png", "jpeg", "webp"]
MODERATION_OPTIONS = ["auto", "low"]
INPUT_FIDELITY_OPTIONS = ["auto", "low", "high"]
JIMENG_MODEL_OPTIONS = [
    "doubao-seedream-5-0-260128",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
    "doubao-seedream-3-0-t2i-250415",
]
JIMENG_SIZE_OPTIONS = [
    "2K",
    "3K",
    "4K",
    "2048x2048",
    "2304x1728",
    "1728x2304",
    "2848x1600",
    "1600x2848",
    "2496x1664",
    "1664x2496",
    "3136x1344",
    "3072x3072",
    "3456x2592",
    "2592x3456",
    "4096x2304",
    "2304x4096",
    "2496x3744",
    "3744x2496",
    "4704x2016",
    "custom",
]
JIMENG_RESPONSE_FORMAT_OPTIONS = ["url", "b64_json"]
JIMENG_OUTPUT_FORMAT_OPTIONS = ["jpeg", "png"]
JIMENG_SEQUENTIAL_OPTIONS = ["disabled", "auto"]
JIMENG_OPTIMIZE_PROMPT_OPTIONS = ["auto", "standard", "fast", "disabled"]


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
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    key = _load_config_value("api_key")
    if key:
        return key
    raise RuntimeError(
        "No API key found. Either paste your key into the api_key field, "
        f"create {PLUGIN_CONFIG_PATH}, or set OPENAI_API_KEY."
    )


def _load_base_url(base_url_input=""):
    if base_url_input and base_url_input.strip():
        return base_url_input.strip().rstrip("/")

    for env_name in ("OPENAI_BASE_URL", "GPT_IMAGE2_BASE_URL", "MUAPI_BASE_URL"):
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url.rstrip("/")

    base_url = _load_config_value("base_url")
    if base_url:
        return base_url.rstrip("/")

    return DEFAULT_BASE_URL


def _load_jimeng_api_key(api_key_input):
    if api_key_input and api_key_input.strip():
        return api_key_input.strip()

    for env_name in ("JIMENG_API_KEY", "VOLCENGINE_API_KEY", "ARK_API_KEY"):
        env_key = os.environ.get(env_name, "").strip()
        if env_key:
            return env_key

    for key in ("jimeng_api_key", "volcengine_api_key", "ark_api_key"):
        value = _load_config_value(key)
        if value:
            return value

    raise RuntimeError(
        "No Jimeng API key found. Paste it into jimeng_api_key, "
        "set JIMENG_API_KEY/VOLCENGINE_API_KEY/ARK_API_KEY, or add "
        "jimeng_api_key to config.json."
    )


def _load_jimeng_base_url(base_url_input=""):
    if base_url_input and base_url_input.strip():
        return base_url_input.strip().rstrip("/")

    for env_name in ("JIMENG_BASE_URL", "VOLCENGINE_ARK_BASE_URL", "ARK_BASE_URL"):
        env_url = os.environ.get(env_name, "").strip()
        if env_url:
            return env_url.rstrip("/")

    for key in ("jimeng_base_url", "volcengine_base_url", "ark_base_url"):
        value = _load_config_value(key)
        if value:
            return value.rstrip("/")

    return DEFAULT_JIMENG_BASE_URL


def _load_user_agent():
    env_user_agent = os.environ.get("OPENAI_USER_AGENT", "").strip()
    if env_user_agent:
        return env_user_agent

    user_agent = _load_config_value("user_agent")
    if user_agent:
        return user_agent

    return DEFAULT_USER_AGENT


def _auth_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": _load_user_agent(),
    }


def _image_options_payload(
    model,
    n,
    size,
    quality,
    background,
    output_format,
    output_compression,
    moderation,
    user,
    stream,
    partial_images,
    style=None,
    input_fidelity=None,
):
    payload = {
        "model": model,
        "n": int(n),
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "moderation": moderation,
    }
    if output_format in ("jpeg", "webp"):
        payload["output_compression"] = int(output_compression)
    if user and user.strip():
        payload["user"] = user.strip()
    if stream:
        payload["stream"] = True
    if stream and partial_images:
        payload["partial_images"] = int(partial_images)
    if style and style.strip():
        payload["style"] = style.strip()
    if input_fidelity and input_fidelity != "auto" and model != "gpt-image-2":
        payload["input_fidelity"] = input_fidelity
    return payload


def _json_to_image_result(data):
    url = _output_image_url(data)
    return _download_image(url), url


def _read_stream_response(resp):
    _check(resp)
    completed = None
    last_image_event = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", ""))
        if event.get("b64_json"):
            last_image_event = event
        if event_type.endswith(".completed"):
            completed = event
            break
    if completed:
        return completed
    if last_image_event:
        return last_image_event
    raise RuntimeError("Streaming response finished without a generated image.")


def _post_json(api_key, base_url, endpoint, payload):
    use_stream = bool(payload.get("stream"))
    resp = requests.post(
        f"{base_url}/{endpoint}",
        headers={**_auth_headers(api_key), "Content-Type": "application/json"},
        json=payload,
        timeout=600,
        stream=use_stream,
    )
    if use_stream:
        return _read_stream_response(resp), _request_id(resp)
    _check(resp)
    return resp.json(), _request_id(resp)


def _post_multipart(api_key, base_url, endpoint, data, files, stream_response=False):
    resp = requests.post(
        f"{base_url}/{endpoint}",
        headers=_auth_headers(api_key),
        data=data,
        files=files,
        timeout=600,
        stream=stream_response,
    )
    if stream_response:
        return _read_stream_response(resp), _request_id(resp)
    _check(resp)
    return resp.json(), _request_id(resp)


def _read_jimeng_stream_response(resp):
    _check(resp)
    images = []
    usage = {}
    final_event = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if raw == "[DONE]":
            break
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", ""))
        if event.get("usage"):
            usage = event.get("usage") or {}
        if event.get("error"):
            raise RuntimeError(f"Jimeng stream error: {event['error']}")
        if event.get("b64_json") or event.get("url"):
            images.append(event)
        if event_type.endswith(".completed"):
            final_event = event
            break
    if final_event and final_event.get("images"):
        images.extend(final_event["images"])
    if not images and final_event:
        _append_image_url([], final_event)
        images.append(final_event)
    if not images:
        raise RuntimeError("Jimeng streaming response finished without an image.")
    return {"data": images, "usage": usage, "stream_event": final_event or {}}


def _post_jimeng_json(api_key, base_url, payload):
    use_stream = bool(payload.get("stream"))
    resp = requests.post(
        f"{base_url}/images/generations",
        headers={
            **_auth_headers(api_key),
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=900,
        stream=use_stream,
    )
    if use_stream:
        return _read_jimeng_stream_response(resp), _request_id(resp)
    _check(resp)
    return resp.json(), _request_id(resp)


def _request_id(resp):
    return (
        resp.headers.get("x-request-id")
        or resp.headers.get("openai-request-id")
        or ""
    )


def _image_tensor_to_file_tuple(image_tensor, filename):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    buf.seek(0)
    return (filename, buf, "image/png")


def _image_tensor_to_data_url(image_tensor):
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    arr = (image_tensor.cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _multipart_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _output_image_url(result):
    out = result.get("outputs") or result.get("output") or []
    if isinstance(out, list) and out:
        return str(out[0])
    if isinstance(out, str):
        return out
    data = result.get("data")
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            if item.get("url"):
                return str(item["url"])
            if item.get("b64_json"):
                output_format = result.get("output_format") or "png"
                return f"data:image/{output_format};base64,{item['b64_json']}"
    if result.get("b64_json"):
        output_format = result.get("output_format") or "png"
        return f"data:image/{output_format};base64,{result['b64_json']}"
    for k in ("image_url", "url"):
        if result.get(k):
            return str(result[k])
    raise RuntimeError(f"No output image URL in result: {result}")


def _append_image_url(urls, item, default_format="png"):
    if isinstance(item, str) and item:
        urls.append(item)
        return
    if not isinstance(item, dict):
        return
    if item.get("url"):
        urls.append(str(item["url"]))
    elif item.get("image_url"):
        urls.append(str(item["image_url"]))
    elif item.get("b64_json"):
        output_format = item.get("output_format") or default_format
        urls.append(f"data:image/{output_format};base64,{item['b64_json']}")


def _output_image_urls(result):
    output_format = str(result.get("output_format") or "png")
    urls = []
    for key in ("data", "images", "outputs", "output", "choices"):
        value = result.get(key)
        if isinstance(value, list):
            for item in value:
                _append_image_url(urls, item, output_format)
        elif value:
            _append_image_url(urls, value, output_format)
    _append_image_url(urls, result, output_format)
    seen = []
    for url in urls:
        if url not in seen:
            seen.append(url)
    if not seen:
        raise RuntimeError(f"No output image URL in result: {result}")
    return seen


def _download_image(url):
    if url.startswith("data:image/") and ";base64," in url:
        _, b64_data = url.split(";base64,", 1)
        img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    r = requests.get(url, headers={"User-Agent": _load_user_agent()}, timeout=120)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _download_images(urls):
    tensors = [_download_image(url) for url in urls]
    if not tensors:
        raise RuntimeError("No generated images to download.")
    return torch.cat(tensors, dim=0)


def _check(resp):
    if resp.status_code == 401:
        raise RuntimeError("Auth failed — check your API key.")
    if resp.status_code == 402:
        raise RuntimeError("Insufficient credits or billing issue.")
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
    Store your OpenAI API key once and wire it to any GPT-Image-2 node.
    Alternatively, leave all api_key fields empty — nodes auto-read from
    OPENAI_API_KEY, this plugin's config.json, then ~/.muapi/config.json.
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
                        "tooltip": "Your OpenAI API key or OpenAI-compatible provider API key.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(self, api_key):
        return (_load_api_key(api_key),)


class GPTImage2BaseUrl:
    """
    Provide an OpenAI-compatible API base URL.
    Defaults to https://api.openai.com/v1 when left blank.
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
                        "tooltip": "OpenAI-compatible API base URL, for example https://api.example.com/v1",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base_url",)
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(self, base_url):
        return (_load_base_url(base_url),)


class GPTImage2TextToImage:
    """
    GPT-Image-2 Text-to-Image
    --------------------------
    Generate a high-quality image from a text prompt using GPT-Image-2.

    Endpoint: POST /v1/images/generations
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
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "size": (SIZE_OPTIONS, {"default": "auto"}),
                "quality": (QUALITY_OPTIONS, {"default": "auto"}),
                "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
                "output_format": (OUTPUT_FORMAT_OPTIONS, {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "moderation": (MODERATION_OPTIONS, {"default": "auto"}),
                "stream": ("BOOLEAN", {"default": False}),
                "partial_images": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                "style": ("STRING", {"multiline": False, "default": ""}),
                "user": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(
        self,
        prompt,
        api_key="",
        base_url="",
        model="gpt-image-2",
        n=1,
        size="auto",
        quality="auto",
        background="auto",
        output_format="png",
        output_compression=100,
        moderation="auto",
        stream=False,
        partial_images=0,
        style="",
        user="",
    ):
        api_key = _load_api_key(api_key)
        base_url = _load_base_url(base_url)
        payload = {"prompt": prompt}
        payload.update(
            _image_options_payload(
                model=model,
                n=n,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                output_compression=output_compression,
                moderation=moderation,
                user=user,
                stream=stream,
                partial_images=partial_images,
                style=style,
            )
        )
        print(f"[GPTImage2 T2I] Submitting to {base_url}/images/generations...")
        result, request_id = _post_json(api_key, base_url, "images/generations", payload)
        image, url = _json_to_image_result(result)
        request_id = request_id or str(result.get("created", ""))
        print(f"[GPTImage2 T2I] Done -> {request_id or url[:80]}")
        return (image, url, request_id)


class GPTImage2ImageToImage:
    """
    GPT-Image-2 Image-to-Image
    ---------------------------
    Transform or edit up to 16 reference images guided by a text prompt.
    Common uses: style transfer, product shots, scene editing.

    Endpoint: POST /v1/images/edits

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
                "model": (MODEL_OPTIONS, {"default": "gpt-image-2"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "size": (SIZE_OPTIONS, {"default": "auto"}),
                "quality": (QUALITY_OPTIONS, {"default": "auto"}),
                "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
                "input_fidelity": (INPUT_FIDELITY_OPTIONS, {"default": "auto"}),
                "output_format": (OUTPUT_FORMAT_OPTIONS, {"default": "png"}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "moderation": (MODERATION_OPTIONS, {"default": "auto"}),
                "stream": ("BOOLEAN", {"default": False}),
                "partial_images": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
                "user": ("STRING", {"multiline": False, "default": ""}),
                "mask_image": ("IMAGE",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
                "image_15": ("IMAGE",),
                "image_16": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "request_id")
    FUNCTION = "run"
    CATEGORY = GPT_IMAGE_CATEGORY

    def run(
        self,
        prompt,
        api_key="",
        base_url="",
        model="gpt-image-2",
        n=1,
        size="auto",
        quality="auto",
        background="auto",
        input_fidelity="auto",
        output_format="png",
        output_compression=100,
        moderation="auto",
        stream=False,
        partial_images=0,
        user="",
        mask_image=None,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        image_11=None,
        image_12=None,
        image_13=None,
        image_14=None,
        image_15=None,
        image_16=None,
    ):
        api_key = _load_api_key(api_key)
        base_url = _load_base_url(base_url)
        tensors = [
            image_1, image_2, image_3, image_4, image_5, image_6, image_7,
            image_8, image_9, image_10, image_11, image_12, image_13,
            image_14, image_15, image_16,
        ]
        files = []
        for i, img in enumerate(tensors, 1):
            if img is not None:
                files.append(("image[]", _image_tensor_to_file_tuple(img, f"image_{i}.png")))
        if not files:
            raise ValueError("At least one input image is required.")

        payload = {"prompt": prompt}
        payload.update(
            _image_options_payload(
                model=model,
                n=n,
                size=size,
                quality=quality,
                background=background,
                output_format=output_format,
                output_compression=output_compression,
                moderation=moderation,
                user=user,
                stream=stream,
                partial_images=partial_images,
                input_fidelity=input_fidelity,
            )
        )
        if mask_image is not None:
            files.append(("mask", _image_tensor_to_file_tuple(mask_image, "mask.png")))

        data = {key: _multipart_value(value) for key, value in payload.items()}
        print(f"[GPTImage2 I2I] Submitting ({len(files)} file(s)) to {base_url}/images/edits...")
        result, request_id = _post_multipart(
            api_key,
            base_url,
            "images/edits",
            data,
            files,
            stream_response=bool(payload.get("stream")),
        )
        image, url = _json_to_image_result(result)
        request_id = request_id or str(result.get("created_at") or result.get("created") or "")
        print(f"[GPTImage2 I2I] Done -> {request_id or url[:80]}")
        return (image, url, request_id)


class JimengApiKey:
    """
    Store a Volcano Ark/Jimeng API key once and wire it to Jimeng nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "jimeng_api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Volcano Ark API key for Seedream/Jimeng models.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("jimeng_api_key",)
    FUNCTION = "run"
    CATEGORY = JIMENG_CATEGORY

    def run(self, jimeng_api_key):
        return (_load_jimeng_api_key(jimeng_api_key),)


class JimengBaseUrl:
    """
    Provide a Volcano Ark-compatible base URL.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_JIMENG_BASE_URL,
                        "tooltip": "Default: https://ark.cn-beijing.volces.com/api/v3",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("base_url",)
    FUNCTION = "run"
    CATEGORY = JIMENG_CATEGORY

    def run(self, base_url):
        return (_load_jimeng_base_url(base_url),)


class JimengSeedreamImage:
    """
    Jimeng/Seedream image generation via Volcano Ark Image generation API.
    Exposes the API parameters documented by Volcano Engine instead of hiding
    them behind presets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for generation or editing.",
                    },
                ),
            },
            "optional": {
                "jimeng_api_key": ("STRING", {"multiline": False, "default": ""}),
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "model": (JIMENG_MODEL_OPTIONS, {"default": "doubao-seedream-5-0-260128"}),
                "size": (JIMENG_SIZE_OPTIONS, {"default": "2K"}),
                "custom_width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "custom_height": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "response_format": (JIMENG_RESPONSE_FORMAT_OPTIONS, {"default": "b64_json"}),
                "output_format": (JIMENG_OUTPUT_FORMAT_OPTIONS, {"default": "jpeg"}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647, "step": 1}),
                "stream": ("BOOLEAN", {"default": False}),
                "sequential_image_generation": (
                    JIMENG_SEQUENTIAL_OPTIONS,
                    {"default": "disabled"},
                ),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "enable_web_search": ("BOOLEAN", {"default": False}),
                "optimize_prompt_mode": (
                    JIMENG_OPTIMIZE_PROMPT_OPTIONS,
                    {"default": "auto"},
                ),
                "extra_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Optional raw JSON merged into the request payload.",
                    },
                ),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "image_11": ("IMAGE",),
                "image_12": ("IMAGE",),
                "image_13": ("IMAGE",),
                "image_14": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "request_id", "raw_response")
    FUNCTION = "run"
    CATEGORY = JIMENG_CATEGORY

    def run(
        self,
        prompt,
        jimeng_api_key="",
        base_url="",
        model="doubao-seedream-5-0-260128",
        size="2K",
        custom_width=2048,
        custom_height=2048,
        response_format="b64_json",
        output_format="jpeg",
        watermark=False,
        seed=-1,
        stream=False,
        sequential_image_generation="disabled",
        max_images=1,
        enable_web_search=False,
        optimize_prompt_mode="auto",
        extra_json="",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
        image_9=None,
        image_10=None,
        image_11=None,
        image_12=None,
        image_13=None,
        image_14=None,
    ):
        api_key = _load_jimeng_api_key(jimeng_api_key)
        base_url = _load_jimeng_base_url(base_url)

        size_value = (
            f"{int(custom_width)}x{int(custom_height)}"
            if size == "custom"
            else size.split(" ", 1)[0]
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "size": size_value,
            "response_format": response_format,
            "output_format": output_format,
            "watermark": bool(watermark),
            "stream": bool(stream),
        }
        if int(seed) >= 0:
            payload["seed"] = int(seed)

        tensors = [
            image_1, image_2, image_3, image_4, image_5, image_6, image_7,
            image_8, image_9, image_10, image_11, image_12, image_13, image_14,
        ]
        image_inputs = [_image_tensor_to_data_url(img) for img in tensors if img is not None]
        if image_inputs:
            payload["image"] = image_inputs[0] if len(image_inputs) == 1 else image_inputs

        if sequential_image_generation != "disabled":
            payload["sequential_image_generation"] = sequential_image_generation
            payload["sequential_image_generation_options"] = {
                "max_images": int(max_images),
            }

        if enable_web_search:
            payload["tools"] = [{"type": "web_search"}]

        if optimize_prompt_mode != "disabled":
            mode = "standard" if optimize_prompt_mode == "auto" else optimize_prompt_mode
            payload["optimize_prompt_options"] = {"mode": mode}

        if extra_json and extra_json.strip():
            try:
                extra_payload = json.loads(extra_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"extra_json is not valid JSON: {exc}") from exc
            if not isinstance(extra_payload, dict):
                raise ValueError("extra_json must decode to a JSON object.")
            payload.update(extra_payload)

        if payload.get("sequential_image_generation") == "auto":
            requested_total = len(image_inputs) + int(
                payload.get("sequential_image_generation_options", {}).get("max_images", 1)
            )
            if requested_total > 15:
                raise ValueError(
                    "Jimeng group generation supports reference images + generated images <= 15."
                )

        print(f"[Jimeng Seedream] Submitting to {base_url}/images/generations...")
        result, request_id = _post_jimeng_json(api_key, base_url, payload)
        urls = _output_image_urls(result)
        images = _download_images(urls)
        raw_response = json.dumps(result, ensure_ascii=False)
        request_id = request_id or str(result.get("id") or result.get("created") or "")
        print(f"[Jimeng Seedream] Done -> {len(urls)} image(s)")
        return (images, "\n".join(urls), request_id, raw_response)


NODE_CLASS_MAPPINGS = {
    "GPTImage2ApiKey":        GPTImage2ApiKey,
    "GPTImage2BaseUrl":       GPTImage2BaseUrl,
    "GPTImage2TextToImage":   GPTImage2TextToImage,
    "GPTImage2ImageToImage":  GPTImage2ImageToImage,
    "JimengApiKey":           JimengApiKey,
    "JimengBaseUrl":          JimengBaseUrl,
    "JimengSeedreamImage":    JimengSeedreamImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2ApiKey":        "🔑 GPT-Image-2 API Key",
    "GPTImage2BaseUrl":       "🌐 GPT-Image-2 Base URL",
    "GPTImage2TextToImage":   "🖼️ GPT-Image-2 Text to Image",
    "GPTImage2ImageToImage":  "🖼️ GPT-Image-2 Image to Image",
    "JimengApiKey":           "🔑 Jimeng API Key",
    "JimengBaseUrl":          "🌐 Jimeng Base URL",
    "JimengSeedreamImage":    "🖼️ Jimeng Seedream Image",
}
