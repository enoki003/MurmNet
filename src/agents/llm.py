"""
Language model loader and wrapper.
Handles loading and inference with small language models.
"""

import json
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.config import config, ModelQuantization


class LanguageModel:
    """
    Wrapper for language models.
    Handles loading, inference, and resource management.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantization: Optional[ModelQuantization] = None,
    ):
        """
        Initialize language model.
        
        Args:
            model_name: Name/path of the model
            device: Device to run on
            quantization: Quantization mode
        """
        self.model_name = model_name or config.model.default_model_name
        self.device = device or config.model.model_device.value
        self.quantization = quantization or config.model.model_quantization
        
        # Check device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration."""
        if self.quantization == ModelQuantization.INT8:
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == ModelQuantization.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        return None
    
    def _load_model(self) -> None:
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(config.model.model_cache_dir),
                trust_remote_code=True,
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get quantization config
            quantization_config = self._get_quantization_config()
            
            # Detect model type and load appropriately
            # T5 models use Seq2SeqLM, most others use CausalLM
            try:
                if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                    logger.info("Loading as Seq2Seq model (T5/FLAN)")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        cache_dir=str(config.model.model_cache_dir),
                        quantization_config=quantization_config,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    )
                else:
                    logger.info("Loading as Causal LM")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=str(config.model.model_cache_dir),
                        quantization_config=quantization_config,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    )
            except Exception as model_error:
                logger.warning(f"Failed with detected type, trying alternative: {model_error}")
                # Try the other type
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        cache_dir=str(config.model.model_cache_dir),
                        quantization_config=quantization_config,
                        device_map="auto" if self.device == "cuda" else None,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    )
                except:
                    # If that fails too, raise the original error
                    raise model_error
            
            # Move to device if not using auto device mapping
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(
                f"Model loaded successfully on {self.device} "
                f"(quantization: {self.quantization.value})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )
            
            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break
            
            return generated_text.strip()
            
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
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
        
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded")
