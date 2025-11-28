import os
import json
import base64
import io
import warnings
import requests
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
from urllib.parse import urlparse
from dotenv import load_dotenv
from anthropic import Anthropic
import openai
from openai import OpenAI

# Try to import PIL for image resizing
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


class LLMClient:
    """
    统一的大模型客户端类，支持多种LLM服务提供商
    """

    # # 支持的提供商列表
    # PROVIDERS = ["openai", "claude", "deepseek", "openrouter", "xai"]
    # # 默认模型映射
    # DEFAULT_MODELS = {
    #     "openrouter": "anthropic/claude-3.7-sonnet", # openrouter/auto
    #     "openai": "gpt-4o",
    #     "claude": "claude-3-haiku-20240307",
    #     "deepseek": "deepseek-chat-v3-0324",
    #     "xai": "grok-2-latest"
    # }
    # # API端点映射
    # API_ENDPOINTS = {
    #     "openai": None,  # 使用SDK
    #     "claude": None,  # 使用SDK
    #     "deepseek": "https://api.deepseek.com/chat/completions",
    #     "openrouter": "https://openrouter.ai/api/v1",
    #     "xai": "https://api.x.ai/v1"  # 更新为正确的 XAI API 端点
    # }
    # # 环境变量密钥映射
    # ENV_KEYS = {
    #     "openai": "OPENAI_API_KEY",
    #     "claude": "ANTHROPIC_API_KEY",
    #     "deepseek": "DEEPSEEK_API_KEY",
    #     "openrouter": "OPENROUTER_API_KEY",
    #     "xai": "XAI_API_KEY"
    # }

    PROVIDERS = ["openrouter"]

    MODELS = [
        "deepseek/deepseek-chat-v3-0324",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4",
        "anthropic/claude-opus-4.1",
        "openai/o3-mini",
        "openai/o4-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/o3",
        "openai/gpt-5-chat",
        "openai/gpt-5.1",
        "google/gemini-2.5",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash-preview-image",
        "google/gemini-3-pro-image-preview",
        "google/gemini-3-pro-preview",
        "deepseek/deepseek-chat-v3.1",
        "x-ai/grok-4",
        "x-ai/grok-4.1-fast",
        "qwen/qwen3-max",
        "moonshotai/kimi-k2-0905",
        "z-ai/glm-4.6",
    ]

    DEFAULT_MODELS = {"openrouter": "deepseek/deepseek-chat-v3-0324"}

    if os.getenv("AGENT_ACCESS_TOKEN"):
        # API端点映射
        API_ENDPOINTS = {"openrouter": os.getenv("AGENTICS_LLM_URL")}
        # 环境变量密钥映射
        ENV_KEYS = {"openrouter": "AGENT_ACCESS_TOKEN"}
    else:
        API_ENDPOINTS = {"openrouter": "https://openrouter.ai/api/v1"}
        ENV_KEYS = {"openrouter": "OPENROUTER_API_KEY"}

    def __init__(self, provider: str = "openrouter", model: Optional[str] = None):
        """
        初始化LLM客户端

        Args:
            provider: 提供商名称，支持 "openai", "claude", "deepseek", "openrouter", "xai"
            model: 模型名称，如果为None则使用默认模型
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: {', '.join(self.PROVIDERS)}"
            )

        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]

        if self.model not in self.MODELS:
            raise ValueError(
                f"Unsupported model: {self.model}. Supported models: {', '.join(self.MODELS)}"
            )

        # 加载环境变量
        load_dotenv()
        self.api_key = os.getenv(self.ENV_KEYS[provider]) or os.environ.get(
            self.ENV_KEYS[provider]
        )

        if not self.api_key:
            raise ValueError(
                f"{self.ENV_KEYS[provider]} not found in environment variables"
            )

        # 初始化客户端
        self.client = None
        if provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        elif provider == "claude":
            self.client = Anthropic(api_key=self.api_key)
        elif provider == "xai":
            self.client = OpenAI(
                api_key=self.api_key, base_url=self.API_ENDPOINTS[provider]
            )
        elif provider == "openrouter":
            self.client = OpenAI(
                api_key=self.api_key, base_url=self.API_ENDPOINTS[provider]
            )

    def close(self):
        """关闭客户端连接，释放资源"""
        if self.client and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception as e:
                # 静默处理关闭错误，不影响主流程
                pass

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭"""
        self.close()
        return False

    def _process_input_image(self, img_input: Union[str, bytes]) -> str:
        """处理输入图片，转换为 data URL 格式。

        Args:
            img_input: 图片输入，可以是：
                - HTTP/HTTPS URL（str，如 "https://example.com/image.png"）
                - 文件路径（str）
                - base64 字符串（str，以 data:image/ 开头或纯 base64）
                - 图片字节数据（bytes）

        Returns:
            data URL 格式的字符串，如 "data:image/png;base64,..."

        Raises:
            ValueError: 如果图片格式无效或文件不存在
            requests.RequestException: 如果从 URL 下载图片失败
        """
        # Handle file path
        if isinstance(img_input, str):
            # Check if it's already a data URL
            if img_input.startswith("data:image/"):
                return img_input

            # Check if it's an HTTP/HTTPS URL
            if img_input.startswith("http://") or img_input.startswith("https://"):
                try:
                    # Download image from URL
                    response = requests.get(img_input, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content

                    # Detect MIME type from Content-Type header or URL extension
                    content_type = response.headers.get("Content-Type", "")
                    if content_type and content_type.startswith("image/"):
                        mime_type = content_type
                    else:
                        # Fallback to URL extension
                        parsed_url = urlparse(img_input)
                        ext = Path(parsed_url.path).suffix.lower()
                        mime_types = {
                            ".png": "image/png",
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".gif": "image/gif",
                            ".webp": "image/webp",
                        }
                        mime_type = mime_types.get(ext, "image/png")

                    b64_str = base64.b64encode(image_bytes).decode("utf-8")
                    return f"data:{mime_type};base64,{b64_str}"
                except requests.RequestException as e:
                    raise ValueError(
                        f"Failed to download image from URL {img_input}: {e}"
                    )

            # Check if it's a base64 string (without data URL prefix)
            # Try to detect if it's base64 by checking if it's a valid path
            img_path = Path(img_input)
            if img_path.exists() and img_path.is_file():
                # It's a file path, read and encode
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                # Detect MIME type from file extension
                ext = img_path.suffix.lower()
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                mime_type = mime_types.get(ext, "image/png")
                b64_str = base64.b64encode(image_bytes).decode("utf-8")
                return f"data:{mime_type};base64,{b64_str}"
            else:
                # Assume it's a base64 string
                # Try to decode to verify it's valid base64
                try:
                    base64.b64decode(img_input, validate=True)
                    # If it doesn't have data URL prefix, add it
                    if not img_input.startswith("data:"):
                        return f"data:image/png;base64,{img_input}"
                    return img_input
                except Exception as e:
                    # Provide more helpful error message
                    preview = (
                        str(img_input)[:50] + "..."
                        if len(str(img_input)) > 50
                        else str(img_input)
                    )
                    raise ValueError(
                        f"Invalid image input: {preview}. "
                        f"Expected: HTTP/HTTPS URL, file path, base64 string, or data URL. "
                        f"Error: {type(e).__name__}: {str(e)}"
                    )

        # Handle bytes
        elif isinstance(img_input, bytes):
            b64_str = base64.b64encode(img_input).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"

        else:
            raise ValueError(
                f"Unsupported image input type: {type(img_input)}. Must be str (path/base64) or bytes."
            )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 10000,
        timeout: int = 180,
    ) -> Dict[str, Any]:
        """
        通用的聊天完成方法

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "Hello"}, ...]
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)，默认180秒

        Returns:
            响应结果
        """
        if self.provider in ["openai", "xai", "openrouter"]:
            return self._openai_completion(messages, temperature, max_tokens, timeout)
        elif self.provider == "claude":
            return self._claude_completion(messages, temperature, max_tokens, timeout)
        else:
            return self._api_completion(messages, temperature, max_tokens, timeout)

    def _openai_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> Dict[str, Any]:
        """OpenAI API调用"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # 转换为统一格式
        return {
            "content": response.choices[0].message.content,
            "raw_response": response,
            "finish_reason": response.choices[0].finish_reason,
        }

    def _claude_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> Dict[str, Any]:
        """Claude API调用"""
        # 提取system消息
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append(msg)

        # 调用Claude API
        response = self.client.messages.create(
            model=self.model,
            system=system_message,  # 系统提示作为单独参数
            messages=claude_messages,  # 不包含system角色的消息
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # 转换为统一格式
        return {
            "content": response.content[0].text,
            "raw_response": response,
            "finish_reason": response.stop_reason,
        }

    def _api_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> Dict[str, Any]:
        """通用API调用（用于DeepSeek、OpenRouter和XAI）"""
        url = self.API_ENDPOINTS[self.provider]

        headers = {"Content-Type": "application/json"}

        # 根据不同提供商设置不同的认证头
        if self.provider == "deepseek":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "openrouter":
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = "https://your-app-url.com"  # OpenRouter需要
            headers["X-Title"] = "Your App Name"  # OpenRouter需要
        elif self.provider == "xai":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()

        # 转换为统一格式
        return {
            "content": result["choices"][0]["message"]["content"],
            "raw_response": result,
            "finish_reason": result["choices"][0].get("finish_reason"),
        }

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        timeout: int = 180,
        input_images: Optional[List[Union[str, bytes]]] = None,
        auto_resize: bool = True,
    ) -> Dict[str, Any]:
        """生成图片（使用OpenAI兼容SDK）。
        返回包含base64字符串的统一结构。

        Note: OpenRouter uses /chat/completions with modalities for image generation,
        so we use chat.completions.create() instead of images.generate().

        支持的模型包括：
        - google/gemini-2.5-flash-preview-image
        - google/gemini-3-pro-image-preview

        Args:
            prompt: 图像生成的提示词
            size: 图像尺寸，格式为 "WIDTHxHEIGHT"，默认 "1024x1024"
                  注意：某些模型可能只支持特定尺寸（如 1024x1024 正方形），
                  非标准尺寸可能无法正常生成。建议使用 1024x1024、512x512 等标准尺寸。
            timeout: 请求超时时间(秒)，默认180秒
            input_images: 可选的输入图片列表，支持以下格式：
                - HTTP/HTTPS URL：["https://example.com/image.png"]
                - 文件路径列表：["path/to/image1.png", "path/to/image2.png"]
                - base64字符串列表：["base64_string1", "base64_string2"]
                - data URL：["data:image/png;base64,..."]
                - 混合格式（URL、路径、base64等）
                这些图片将作为参考图片传递给模型
            auto_resize: 如果为 True，生成后自动调整图片到指定尺寸（需要 PIL/Pillow）
                        这对于非正方形尺寸特别有用，因为模型可能只生成正方形图片
                        如果 PIL 不可用，此参数将被忽略

        Returns:
            包含base64图像数据的字典，格式为 {"b64": str, "raw_response": Any}

        Raises:
            ValueError: 如果响应中缺少图像数据，或输入图片格式无效
        """
        # Prepare request parameters
        # Note: Size adjustment is handled by post-processing (auto_resize),
        # so we don't modify the prompt with size instructions
        enhanced_prompt = prompt

        # Build message content - support multiple input images
        message_content = []

        # Add text prompt
        message_content.append({"type": "text", "text": enhanced_prompt})

        # Add input images if provided
        if input_images:
            # Filter out None and empty values
            valid_images = [img for img in input_images if img]
            if not valid_images:
                raise ValueError(
                    "input_images list contains no valid images (all None or empty)"
                )

            for img_input in valid_images:
                try:
                    image_url = self._process_input_image(img_input)
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to process input image '{img_input}': {e}"
                    ) from e

        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "modalities": ["image", "text"],
            "timeout": timeout,
        }

        # Use chat completions with modalities for image generation
        # Note: Size adjustment is handled by post-processing (auto_resize),
        # so we don't pass size/aspect_ratio parameters to the API
        response = self.client.chat.completions.create(**request_params)

        # Extract image data from response
        b64_data = None
        choices = response.choices if hasattr(response, "choices") else []

        if choices and len(choices) > 0:
            choice = choices[0]
            message = choice.message if hasattr(choice, "message") else {}

            # Convert Pydantic model to dict for easier access
            message_dict = None
            if hasattr(message, "model_dump"):
                message_dict = message.model_dump()
            elif isinstance(message, dict):
                message_dict = message

            # Check for images field (OpenRouter format) - use dict if available
            if message_dict and "images" in message_dict and message_dict["images"]:
                images_list = message_dict["images"]
                if isinstance(images_list, list) and len(images_list) > 0:
                    first_image = images_list[0]
                    if isinstance(first_image, dict) and "image_url" in first_image:
                        image_url_obj = first_image["image_url"]
                        if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                            url = image_url_obj["url"]
                            if url and url.startswith("data:image/"):
                                b64_data = url.split(",", 1)[1] if "," in url else url

            # Fallback: try attribute access for Pydantic models
            if not b64_data and hasattr(message, "images") and message.images:
                images_list = message.images
                if isinstance(images_list, list) and len(images_list) > 0:
                    first_image = images_list[0]
                    if hasattr(first_image, "image_url"):
                        image_url_obj = first_image.image_url
                        if hasattr(image_url_obj, "url"):
                            url = image_url_obj.url
                            if url and url.startswith("data:image/"):
                                b64_data = url.split(",", 1)[1] if "," in url else url

                    # Also try to convert to dict
                    if not b64_data and hasattr(first_image, "model_dump"):
                        img_dict = first_image.model_dump()
                        if "image_url" in img_dict:
                            url_obj = img_dict["image_url"]
                            if isinstance(url_obj, dict) and "url" in url_obj:
                                url = url_obj["url"]
                                if url.startswith("data:image/"):
                                    b64_data = (
                                        url.split(",", 1)[1] if "," in url else url
                                    )

        if not b64_data:
            # Provide more helpful error message
            error_msg = (
                f"Image generation failed: No image data found in response. "
                f"Model: {self.model}. "
                f"This model may not support image generation, or the API response format may have changed."
            )
            raise ValueError(error_msg)

        # Post-process: Resize image if auto_resize is enabled and size is specified
        if auto_resize and size and PIL_AVAILABLE:
            try:
                # Parse target size
                width, height = size.split("x")
                target_width = int(width)
                target_height = int(height)

                # Decode image
                image_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(image_bytes))

                # Check if resizing is needed
                current_size = img.size
                if current_size != (target_width, target_height):
                    # Resize to target dimensions using high-quality resampling
                    img_resized = img.resize(
                        (target_width, target_height), Image.Resampling.LANCZOS
                    )

                    # Convert back to base64
                    output_buffer = io.BytesIO()
                    # Preserve original format if possible, otherwise use PNG
                    img_format = img.format if img.format else "PNG"
                    img_resized.save(output_buffer, format=img_format)
                    b64_data = base64.b64encode(output_buffer.getvalue()).decode(
                        "utf-8"
                    )
            except Exception as e:
                # If resizing fails, log warning but return original image
                warnings.warn(
                    f"Failed to resize image to {size}: {e}. Returning original image.",
                    UserWarning,
                    stacklevel=2,
                )

        return {"b64": b64_data, "raw_response": response}
