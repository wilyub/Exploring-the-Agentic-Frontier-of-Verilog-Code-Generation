# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPT and Gemini model instance implementations for CVDP benchmark.

- GPT: Uses OpenAI's Responses API for GPT-5.2 and similar models.
- Gemini: Uses Google's Generative AI SDK for gemini-2.5-pro, gemini-3-pro-preview, etc.
"""

import os
import logging
import re
import json
import sys
from typing import Optional, Any

try:
    import openai
except ImportError:
    raise ImportError("The 'openai' package is required for GPT. Install it with 'pip install openai'.")

try:
    import google.genai as genai
except ImportError:
    genai = None  # Optional; required only for Gemini_Instance

from src.config_manager import config

# Import ModelHelpers - used for processing responses
try:
    from src.model_helpers import ModelHelpers
except ImportError:
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.model_helpers import ModelHelpers
    except (ImportError, NameError):
        class ModelHelpers:
            def create_system_prompt(self, base_context, schema=None, category=None):
                context = base_context if base_context is not None else ""
                if schema is not None:
                    if isinstance(schema, list):
                        context += f"\nProvide the response in one of the following JSON schemas: \n"
                        schemas = []
                        for sch in schema:
                            schemas.append(f"{sch}")
                        context += "\nor\n".join(schemas)
                    else:
                        context += f"\nProvide the response in the following JSON schema: {schema}"
                    context += "\nThe response should be in JSON format, including double-quotes around keys and values, and proper escaping of quotes within values, and escaping of newlines."
                return context
                
            def parse_model_response(self, content, files=None, expected_single_file=False):
                if expected_single_file:
                    return content
                return content
                
            def fix_json_formatting(self, content):
                try:
                    content = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', content)
                    content = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9])(\s*[,}])', r': "\1"\2', content)
                    try:
                        json.loads(content)
                    except json.JSONDecodeError:
                        pass
                except:
                    pass
                return content

logging.basicConfig(level=logging.INFO)


class GPT_Instance:
    """
    OpenAI GPT model instance using the Responses API.
    
    Supports GPT-5.2 and other OpenAI models via the Responses API.
    """

    def __init__(self, context: Any = "You are a helpful assistant.", key: Optional[str] = None, model: str = "gpt-5.2"):
        """
        Initialize a GPT model instance.
        
        Args:
            context: The system prompt or context for the model
            key: OpenAI API key (will fall back to OPENAI_API_KEY from config)
            model: The GPT model version to use (default: gpt-5.2)
        """
        self.context = context
        self.model = model
        self.debug = False
        
        api_key = config.get("OPENAI_API_KEY")
        
        if (key is None) and (api_key is None):
            raise ValueError("Unable to create GPT Model - No API key provided. Set OPENAI_API_KEY in environment or config.")
            
        actual_key = key if key is not None else api_key
            
        self.client = openai.Client(
            api_key=actual_key,
            max_retries=0
        )
        
        logging.info(f"Created GPT Model instance. Using model: {self.model}")
    
    def set_debug(self, debug: bool = True) -> None:
        """Enable or disable debug mode."""
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")
    
    def prompt(self, prompt: str, schema: Optional[str] = None, prompt_log: str = "", 
               files: Optional[list] = None, timeout: int = 60, category: Optional[int] = None) -> str:
        """
        Send a prompt to the GPT model and get a response.
        
        Args:
            prompt: The user prompt/query
            schema: Optional JSON schema for structured output
            prompt_log: Path to log the prompt (if not empty)
            files: List of expected output files (if any)
            timeout: Timeout in seconds for the API call (default: 60)
            category: Optional integer indicating the category/problem ID
            
        Returns:
            The model's response as text, or None on error
        """
        if not hasattr(self, 'client'):
            raise ValueError("GPT client not initialized")
            
        helper = ModelHelpers()
        system_prompt = helper.create_system_prompt(self.context, schema, category)
        
        if timeout == 60:
            timeout = config.get("MODEL_TIMEOUT", 60)
            
        expected_single_file = files and len(files) == 1 and schema is None

        if self.debug:
            logging.debug(f"Model: {self.model}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {prompt}")
            logging.debug(f"Timeout: {timeout}s")
            if files:
                logging.debug(f"Expected files: {files}")
            
        if prompt_log:
            try:
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(system_prompt + "\n\n" + "-"*40 + "\n" + prompt)
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log: {e}")
        
        try:
            # OpenAI Responses API
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_output_tokens=32768,
                temperature=1,
                timeout=timeout
            )
            
            # Extract text content from Responses API
            content = ""
            try:
                for item in getattr(response, "output", []) or []:
                    if getattr(item, "type", "") == "message":
                        for part in getattr(item, "content", []) or []:
                            if getattr(part, "type", "") == "output_text":
                                content += getattr(part, "text", "")
                if not content:
                    content = getattr(response, "output_text", "") or ""
            except Exception:
                content = getattr(response, "output_text", "") or ""

            content = (content or "").strip()
            
            if self.debug:
                logging.debug(f"Response content: {content[:500]}...")

            if expected_single_file:
                pass
            elif schema is not None and content.startswith('{') and content.endswith('}'):
                content = helper.fix_json_formatting(content)

            return helper.parse_model_response(content, files, expected_single_file)

        except Exception as e:
            logging.error(f"Error in GPT prompt: {e}")
            return None


class Gemini_Instance:
    """
    Google Gemini model instance using the Generative AI SDK.

    Supports gemini-2.5-pro, gemini-3-pro-preview, and other Gemini models.
    """

    def __init__(self, context: Any = "You are a helpful assistant.", key: Optional[str] = None, model: str = "gemini-2.5-pro"):
        """
        Initialize a Gemini model instance.

        Args:
            context: The system prompt or context for the model
            key: Google API key (will fall back to GEMINI_API_KEY from config/env)
            model: The Gemini model name (e.g. gemini-2.5-pro, gemini-3-pro-preview)
        """
        if genai is None:
            raise ImportError("The 'google-genai' package is required for Gemini. Install it with 'pip install google-genai'.")

        self.context = context
        self.model = model
        self.debug = False

        api_key = config.get("GEMINI_API_KEY")

        if (key is None) and (api_key is None):
            raise ValueError(
                "Unable to create Gemini Model - No API key provided. Set GEMINI_API_KEY in environment or .env."
            )

        self._api_key = key if key is not None else api_key
        self.client = genai.Client(api_key=self._api_key)

        logging.info(f"Created Gemini Model instance. Using model: {self.model}")

    def set_debug(self, debug: bool = True) -> None:
        """Enable or disable debug mode."""
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")

    def prompt(
        self,
        prompt: str,
        schema: Optional[str] = None,
        prompt_log: str = "",
        files: Optional[list] = None,
        timeout: int = 60,
        category: Optional[int] = None,
    ) -> Optional[str]:
        """
        Send a prompt to the Gemini model and get a response.

        Args:
            prompt: The user prompt/query
            schema: Optional JSON schema for structured output
            prompt_log: Path to log the prompt (if not empty)
            files: List of expected output files (if any)
            timeout: Timeout in seconds (Gemini SDK may not support per-call timeout; reserved for compatibility)
            category: Optional integer indicating the category/problem ID

        Returns:
            The model's response as text, or None on error
        """
        if not hasattr(self, "client"):
            raise ValueError("Gemini client not initialized")

        helper = ModelHelpers()
        system_prompt = helper.create_system_prompt(self.context, schema, category)

        if timeout == 60:
            timeout = config.get("MODEL_TIMEOUT", 60)

        expected_single_file = files and len(files) == 1 and schema is None

        if self.debug:
            logging.debug(f"Model: {self.model}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {prompt}")
            logging.debug(f"Timeout: {timeout}s")
            if files:
                logging.debug(f"Expected files: {files}")

        if prompt_log:
            try:
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(system_prompt + "\n\n" + "-" * 40 + "\n" + prompt)
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log: {e}")
        full_prompt = system_prompt + "\n\n" + prompt

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=32768,
                ),
            )

            content = (getattr(response, "text", None) or "").strip()

            if self.debug:
                logging.debug(f"Response content: {content[:500]}...")

            # Log raw model response for inspection and comparison with generate_responses.py
            if prompt_log:
                try:
                    response_log = (
                        prompt_log.replace(".md", "_response.txt")
                        if prompt_log.endswith(".md")
                        else prompt_log + "_response.txt"
                    )
                    with open(response_log, "w+", encoding="utf-8") as f:
                        f.write(content)
                except Exception as e:
                    logging.error(f"Failed to write response log: {e}")

            if expected_single_file:
                pass
            elif schema is not None and content.startswith("{") and content.endswith("}"):
                content = helper.fix_json_formatting(content)

            return helper.parse_model_response(content, files, expected_single_file)

        except Exception as e:
            logging.error(f"Error in Gemini prompt: {e}")
            return None


if __name__ == "__main__":
    # Test the GPT instance
    gpt = GPT_Instance(model="gpt-5.2")
    result = gpt.prompt(
        "Write a simple SystemVerilog module for a 2-to-1 multiplexer with inputs a, b, sel and output y.",
        files=["mux2to1.sv"],
        category=9
    )
    print("=" * 60)
    print("Response:")
    print(result)