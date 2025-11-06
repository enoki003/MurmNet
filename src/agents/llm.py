"""Language model wrapper that calls Ollama via its HTTP API."""

import json
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from src.config import config


class LanguageModel:
    """
    Wrapper for language models.
    Uses Ollama API for inference.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantization: Optional[Any] = None,
    ):
        """
        Initialize language model.
        
        Args:
            model_name: Name/path of the model
            device: Device to run on
            quantization: Quantization mode
        """
        self.model_name = model_name or config.model.ollama_model
        self.base_url = config.model.ollama_base_url.rstrip("/")
        self.timeout = config.model.ollama_timeout_seconds
        self._client = httpx.Client(timeout=self.timeout)
        logger.info(
            "Initialized Ollama client for model %s at %s",
            self.model_name,
            self.base_url,
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens
            if stop_sequences:
                payload["stop"] = stop_sequences

            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            generated_text = data.get("response", "")
            return generated_text.strip()
        except httpx.HTTPError as e:
            logger.error(f"Generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from a list of messages (chat format).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if "message" in data:
                return data["message"].get("content", "").strip()
            return data.get("response", "").strip()
        except httpx.HTTPError as e:
            logger.error(f"Chat generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise
    
    def unload(self) -> None:
        """Cleanup HTTP client."""
        self._client.close()
        logger.info("Ollama client closed")
