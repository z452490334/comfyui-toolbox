# GPT-Image-2 ComfyUI Nodes

> **ComfyUI custom nodes for GPT-Image-2** — OpenAI's latest image generation model.
> Generate and edit images directly inside ComfyUI using the OpenAI Images API or an OpenAI-compatible provider.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom%20Node-blue)](https://github.com/comfyanonymous/ComfyUI)
[![GPT-Image-2](https://img.shields.io/badge/Model-GPT--Image--2-green)](https://platform.openai.com/docs/guides/image-generation)

---

## What is GPT-Image-2?

GPT-Image-2 uses OpenAI-compatible GPT Image models with native image editing support. It supports:

- **Text-to-Image** — generate high-quality images from a text description
- **Image-to-Image** — edit or transform up to 16 reference images guided by a prompt

---

## Nodes

| Node | Description |
|------|-------------|
| 🔑 GPT-Image-2 API Key | Set your key once — wire to all nodes |
| 🌐 GPT-Image-2 Base URL | Set an OpenAI-compatible API base URL for third-party providers |
| 🖼️ GPT-Image-2 Text to Image | Generate an image from a text prompt |
| 🖼️ GPT-Image-2 Image to Image | Edit or transform up to 16 reference images |

---

## Installation

### Via ComfyUI Manager (recommended)
1. Open **ComfyUI Manager** → **Install via Git URL**
2. Paste: `https://github.com/Anil-matcha/gpt-image-2-comfyui`
3. Restart ComfyUI

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Anil-matcha/gpt-image-2-comfyui
pip install -r gpt-image-2-comfyui/requirements.txt
```

---

## Quick Start

1. Create an OpenAI API key, or get a key from an OpenAI-compatible provider
2. Right-click the ComfyUI canvas → **Add Node** → **🖼️ GPT-Image-2**
3. Add a **🔑 GPT-Image-2 API Key** node, paste your key, and wire its output to any generation node
4. Write a prompt and hit **Queue Prompt**

> **Tip:** You can put your key and base URL in this plugin's `config.json` so they are easy to find.
> **Third-party API:** Add a **🌐 GPT-Image-2 Base URL** node or fill the `base_url` field on the generation nodes to use an OpenAI-compatible provider. The default is `https://api.openai.com/v1`.

---

## Node Reference

### 🔑 GPT-Image-2 API Key

Set your OpenAI or OpenAI-compatible API key once and wire the output to all generation nodes. Alternatively, leave every `api_key` field blank — nodes automatically read from `OPENAI_API_KEY`, this plugin's `config.json`, then fall back to `~/.muapi/config.json` for older MuAPI CLI compatibility.

| Field | Description |
|-------|-------------|
| `api_key` | Your OpenAI or OpenAI-compatible provider API key |

**Output:** `api_key` string (wire to generation nodes)

---

### 🌐 GPT-Image-2 Base URL

Set an OpenAI-compatible API base URL once and wire the output to generation nodes. Leave blank to use `https://api.openai.com/v1`.

The easiest persistent setup is to create `config.json` in this plugin directory:

```json
{
  "api_key": "YOUR_KEY",
  "base_url": "https://api.example.com/v1"
}
```

The plugin reads config in this order:

1. Node field input
2. Environment variables
3. This plugin's `config.json`
4. `~/.muapi/config.json`
5. Default OpenAI base URL

You can also configure the base URL with either environment variable:

```bash
OPENAI_BASE_URL=https://api.example.com/v1
GPT_IMAGE2_BASE_URL=https://api.example.com/v1
MUAPI_BASE_URL=https://api.example.com/v1
```

**Output:** `base_url` string (wire to generation nodes)

---

### 🖼️ GPT-Image-2 Text to Image

Generate a high-quality image from a text prompt.

| Field | Description |
|-------|-------------|
| `prompt` | Describe the image you want to generate |
| `api_key` | *(optional)* API key — wire from the API Key node or leave blank |
| `base_url` | *(optional)* OpenAI-compatible API base URL — wire from the Base URL node or leave blank |
| `model` | GPT Image model: `gpt-image-1.5`, `gpt-image-1`, or `gpt-image-1-mini` |
| `n` | Number of images to request, 1-10 |
| `size` | `auto`, `1024x1024`, `1536x1024`, or `1024x1536` |
| `quality` | `auto`, `low`, `medium`, or `high` |
| `background` | `auto`, `transparent`, or `opaque` |
| `output_format` | `png`, `jpeg`, or `webp` |
| `output_compression` | Compression level, 0-100 |
| `moderation` | `auto` or `low` |
| `stream` | Request streaming mode from providers that support it |
| `partial_images` | Number of partial images to request while streaming |
| `style` | Optional DALL-E 3 style passthrough for compatible providers |
| `user` | Optional end-user identifier |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image as a ComfyUI tensor |
| `image_url` | STRING | Generated image URL or data URL |
| `request_id` | STRING | OpenAI request ID or created timestamp |

**Example prompt:**
```
A photorealistic image of a red fox sitting in a snowy forest at dusk.
```

---

### 🖼️ GPT-Image-2 Image to Image

Edit or transform up to 16 reference images guided by a text prompt.

| Field | Description |
|-------|-------------|
| `prompt` | Describe the desired transformation or style |
| `api_key` | *(optional)* API key — wire from the API Key node or leave blank |
| `base_url` | *(optional)* OpenAI-compatible API base URL — wire from the Base URL node or leave blank |
| `model` | GPT Image model: `gpt-image-1.5`, `gpt-image-1`, or `gpt-image-1-mini` |
| `n` | Number of images to request, 1-10 |
| `size` | `auto`, `1024x1024`, `1536x1024`, or `1024x1536` |
| `quality` | `auto`, `low`, `medium`, or `high` |
| `background` | `auto`, `transparent`, or `opaque` |
| `input_fidelity` | `auto`, `low`, or `high` |
| `output_format` | `png`, `jpeg`, or `webp` |
| `output_compression` | Compression level, 0-100 |
| `moderation` | `auto` or `low` |
| `stream` | Request streaming mode from providers that support it |
| `partial_images` | Number of partial images to request while streaming |
| `user` | Optional end-user identifier |
| `mask_image` | Optional mask image for inpainting-compatible providers |
| `image_1` … `image_16` | *(optional)* Reference images to edit |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image as a ComfyUI tensor |
| `image_url` | STRING | Generated image URL or data URL |
| `request_id` | STRING | OpenAI request ID or created timestamp |

**Example prompt:**
```
Transform this product image into a premium e-commerce poster style.
```

---

## API Compatibility

By default, requests go through the OpenAI API. You can override the base URL for any provider that implements the same OpenAI Images API request format:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/images/generations` | POST | Create an image from a prompt |
| `/v1/images/edits` | POST | Edit or generate from one or more input images |

Text-to-image requests are sent as JSON. Image-to-image requests are sent as `multipart/form-data` with `image[]` file fields and an optional `mask` file field.

The generation nodes expose the current OpenAI Images API option names and pass them through in the request payload. The nodes read `data[0].b64_json` responses from GPT Image models, and also support URL responses from compatible providers.

---

## License

MIT — see [LICENSE](LICENSE).
