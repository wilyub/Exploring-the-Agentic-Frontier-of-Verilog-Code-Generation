# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPT, Claude, and Gemini model instance implementations for CVDP benchmark.

- GPT: Uses OpenRouter's OpenAI-compatible API for GPT models (e.g. openai/gpt-4o).
- Claude: Uses OpenRouter's OpenAI-compatible API for Claude models (e.g. anthropic/claude-sonnet-4-5).
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
    raise ImportError("The 'openai' package is required for GPT and Claude. Install it with 'pip install openai'.")

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

# OpenRouter base URL (OpenAI-compatible)
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _build_openrouter_client(api_key: str) -> openai.OpenAI:
    """Return an OpenAI client configured to call OpenRouter."""
    return openai.OpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_BASE_URL,
        max_retries=0,
    )


class GPT_Instance:
    """
    OpenAI GPT model instance routed through OpenRouter.

    Supports GPT models available on OpenRouter (e.g. openai/gpt-4o, openai/gpt-4.1).
    """

    def __init__(self, context: Any = "You are a helpful assistant.", key: Optional[str] = None, model: str = "openai/gpt-4o"):
        """
        Initialize a GPT model instance via OpenRouter.
        
        Args:
            context: The system prompt or context for the model
            key: OpenRouter API key (will fall back to OPENROUTER_API_KEY from config)
            model: The OpenRouter model slug for a GPT model (default: openai/gpt-4o)
        """
        self.context = context
        self.model = model
        self.debug = False
        
        api_key = key or config.get("OPENROUTER_API_KEY")
        
        if api_key is None:
            raise ValueError("Unable to create GPT Model - No API key provided. Set OPENROUTER_API_KEY in environment or config.")
            
        self.client = _build_openrouter_client(api_key)
        logging.info(f"Created GPT Model instance (OpenRouter). Using model: {self.model}")
    
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=1,
                timeout=timeout,
                extra_body={"reasoning": {"effort": "high"}},
            )

            content = (response.choices[0].message.content or "").strip()

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


class Claude_Instance:
    """
    Anthropic Claude model instance routed through OpenRouter.

    Supports Claude models available on OpenRouter (e.g. anthropic/claude-sonnet-4-5,
    anthropic/claude-opus-4-5, anthropic/claude-3-5-haiku).
    """

    def __init__(self, context: Any = "You are a helpful assistant.", key: Optional[str] = None, model: str = "anthropic/claude-sonnet-4-5"):
        """
        Initialize a Claude model instance via OpenRouter.

        Args:
            context: The system prompt or context for the model
            key: OpenRouter API key (will fall back to OPENROUTER_API_KEY from config)
            model: The OpenRouter model slug for a Claude model (default: anthropic/claude-sonnet-4-5)
        """
        self.context = context
        self.model = model
        self.debug = False

        api_key = key or config.get("OPENROUTER_API_KEY")

        if api_key is None:
            raise ValueError(
                "Unable to create Claude Model - No API key provided. Set OPENROUTER_API_KEY in environment or config."
            )

        self.client = _build_openrouter_client(api_key)
        logging.info(f"Created Claude Model instance (OpenRouter). Using model: {self.model}")

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
        Send a prompt to the Claude model via OpenRouter and return the response.

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
        if not hasattr(self, "client"):
            raise ValueError("Claude client not initialized")

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=1,
                timeout=timeout,
                extra_body={"effort": "high"},
            )

            content = (response.choices[0].message.content or "").strip()

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
            logging.error(f"Error in Claude prompt: {e}")
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

        # Build thinking config using thinking_budget (supported in SDK 1.47.0).
        # thinking_level was added in a later SDK version and is not available here.
        # Gemini 3 Pro: set thinking_budget=32768 (max) to approximate "high" reasoning.
        # Gemini 2.5 Pro: leave untouched (dynamic thinking is already the default;
        #   we avoid setting thinking_budget to prevent API errors).
        is_gemini3 = self.model.startswith("gemini-3")
        thinking_config = (
            genai.types.ThinkingConfig(thinking_budget=32768) if is_gemini3 else None
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=32768,
                    thinking_config=thinking_config,
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

class OpenRouter_Instance:
    """
    Generic OpenRouter model instance for third-party models.

    Use this for models that are accessed via OpenRouter but do not belong to
    the OpenAI or Anthropic families, e.g.:
        moonshotai/kimi-k2.5
        minimax/minimax-m2.5
        qwen/qwen3-max-thinking

    Reasoning behaviour is controlled by the reasoning_effort constructor
    parameter:
      - "high" (default): passes extra_body={"reasoning": {"effort": "high"}}
        for models that support OpenRouter's reasoning parameter (kimi, minimax).
      - None: omits the reasoning parameter entirely, for models whose
        thinking is always-on (qwen3-max-thinking).
    """

    # Models whose thinking is always-on; skip the reasoning extra_body.
    _ALWAYS_ON_THINKING = {
        "qwen/qwen3-max-thinking",
    }

    def __init__(
        self,
        context: Any = "You are a helpful assistant.",
        key: Optional[str] = None,
        model: str = "moonshotai/kimi-k2.5",
        reasoning_effort: Optional[str] = "high",
    ):
        self.context = context
        self.model = model
        self.debug = False
        # Auto-suppress reasoning param for always-on thinking models.
        self.reasoning_effort = None if model in self._ALWAYS_ON_THINKING else reasoning_effort
        api_key = key or config.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "Unable to create OpenRouter Model - No API key provided. "
                "Set OPENROUTER_API_KEY in environment or config."
            )
        self.client = _build_openrouter_client(api_key)
        logging.info(f"Created OpenRouter_Instance. Using model: {self.model}")

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
        if not hasattr(self, "client"):
            raise ValueError("OpenRouter client not initialized")
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
        extra_body = (
            {"reasoning": {"effort": self.reasoning_effort}}
            if self.reasoning_effort is not None
            else {}
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=32768,
                temperature=1,
                timeout=timeout,
                extra_body=extra_body,
            )
            content = (response.choices[0].message.content or "").strip()
            if self.debug:
                logging.debug(f"Response content: {content[:500]}...")
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
            logging.error(f"Error in OpenRouter prompt: {e}")
            return None



if __name__ == "__main__":
    TASK = (
        "Write a simple SystemVerilog module for a 2-to-1 multiplexer "
        "with inputs a, b, sel and output y."
    )

    print("=" * 60)
    print("Testing GPT_Instance via OpenRouter ...")
    gpt = GPT_Instance(model="openai/gpt-4o")
    gpt_result = gpt.prompt(TASK, files=["mux2to1.sv"], category=9)
    print("GPT Response:")
    print(gpt_result)

    print("=" * 60)
    print("Testing Claude_Instance via OpenRouter ...")
    claude = Claude_Instance(model="anthropic/claude-sonnet-4-5")
    claude_result = claude.prompt(TASK, files=["mux2to1.sv"], category=9)
    print("Claude Response:")
    print(claude_result)

    print("=" * 60)
    print("Testing Gemini_Instance ...")
    gemini = Gemini_Instance(model="gemini-2.5-pro")
    gemini_result = gemini.prompt(TASK, files=["mux2to1.sv"], category=9)
    print("Gemini Response:")
    print(gemini_result)