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
    Unified LLM client class supporting multiple LLM service providers
    """

    # # List of supported providers
    # PROVIDERS = ["openai", "claude", "deepseek", "openrouter", "xai"]
    # # Default model mapping
    # DEFAULT_MODELS = {
    #     "openrouter": "anthropic/claude-3.7-sonnet", # openrouter/auto
    #     "openai": "gpt-4o",
    #     "claude": "claude-3-haiku-20240307",
    #     "deepseek": "deepseek-chat-v3-0324",
    #     "xai": "grok-2-latest"
    # }
    # # API endpoint mapping
    # API_ENDPOINTS = {
    #     "openai": None,  # Use SDK
    #     "claude": None,  # Use SDK
    #     "deepseek": "https://api.deepseek.com/chat/completions",
    #     "openrouter": "https://openrouter.ai/api/v1",
    #     "xai": "https://api.x.ai/v1"  # Updated to correct XAI API endpoint
    # }
    # # Environment variable key mapping
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
        # API endpoint mapping
        API_ENDPOINTS = {"openrouter": os.getenv("AGENTICS_LLM_URL")}
        # Environment variable key mapping
        ENV_KEYS = {"openrouter": "AGENT_ACCESS_TOKEN"}
    else:
        API_ENDPOINTS = {"openrouter": "https://openrouter.ai/api/v1"}
        ENV_KEYS = {"openrouter": "OPENROUTER_API_KEY"}

    def __init__(self, provider: str = "openrouter", model: Optional[str] = None):
        """
        Initialize LLM client

        Args:
            provider: Provider name, supports "openai", "claude", "deepseek", "openrouter", "xai"
            model: Model name, if None then use default model
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

        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv(self.ENV_KEYS[provider]) or os.environ.get(
            self.ENV_KEYS[provider]
        )

        if not self.api_key:
            raise ValueError(
                f"{self.ENV_KEYS[provider]} not found in environment variables"
            )

        # Initialize client
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
        """Close client connection and release resources"""
        if self.client and hasattr(self.client, "close"):
            try:
                self.client.close()
            except Exception as e:
                # Silently handle close errors, do not affect main flow
                pass

    def __enter__(self):
        """Support context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically close when exiting context"""
        self.close()
        return False

    def _process_input_image(self, img_input: Union[str, bytes]) -> str:
        """Process input image and convert to data URL format.

        Args:
            img_input: Image input, can be:
                - HTTP/HTTPS URL (str, e.g., "https://example.com/image.png")
                - File path (str)
                - base64 string (str, starting with data:image/ or pure base64)
                - Image byte data (bytes)

        Returns:
            Data URL format string, e.g., "data:image/png;base64,..."

        Raises:
            ValueError: If image format is invalid or file does not exist
            requests.RequestException: If downloading image from URL fails
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
        Generic chat completion method

        Args:
            messages: Message list, format: [{"role": "user", "content": "Hello"}, ...]
            temperature: Temperature parameter, controls randomness
            max_tokens: Maximum number of tokens to generate
            timeout: Request timeout in seconds, default 180 seconds

        Returns:
            Response result
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
        """OpenAI API call"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # Convert to unified format
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
        """Claude API call"""
        # Extract system message
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append(msg)

        # Call Claude API
        response = self.client.messages.create(
            model=self.model,
            system=system_message,  # System prompt as separate parameter
            messages=claude_messages,  # Messages without system role
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # Convert to unified format
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
        """Generic API call (for DeepSeek, OpenRouter, and XAI)"""
        url = self.API_ENDPOINTS[self.provider]

        headers = {"Content-Type": "application/json"}

        # Set different authentication headers according to different providers
        if self.provider == "deepseek":
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == "openrouter":
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = "https://your-app-url.com"  # Required by OpenRouter
            headers["X-Title"] = "Your App Name"  # Required by OpenRouter
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

        # Convert to unified format
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
        """Generate image (using OpenAI-compatible SDK).
        Returns a unified structure containing base64 string.

        Note: OpenRouter uses /chat/completions with modalities for image generation,
        so we use chat.completions.create() instead of images.generate().

        Supported models include:
        - google/gemini-2.5-flash-preview-image
        - google/gemini-3-pro-image-preview

        Args:
            prompt: Image generation prompt
            size: Image size, format "WIDTHxHEIGHT", default "1024x1024"
                  Note: Some models may only support specific sizes (e.g., 1024x1024 square),
                  non-standard sizes may not generate properly. Recommended to use standard sizes like 1024x1024, 512x512.
            timeout: Request timeout in seconds, default 180 seconds
            input_images: Optional input image list, supports the following formats:
                - HTTP/HTTPS URL: ["https://example.com/image.png"]
                - File path list: ["path/to/image1.png", "path/to/image2.png"]
                - base64 string list: ["base64_string1", "base64_string2"]
                - data URL: ["data:image/png;base64,..."]
                - Mixed formats (URL, path, base64, etc.)
                These images will be passed to the model as reference images
            auto_resize: If True, automatically resize image to specified size after generation (requires PIL/Pillow)
                        This is particularly useful for non-square sizes, as models may only generate square images
                        If PIL is not available, this parameter will be ignored

        Returns:
            Dictionary containing base64 image data, format: {"b64": str, "raw_response": Any}

        Raises:
            ValueError: If image data is missing in response, or input image format is invalid
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
