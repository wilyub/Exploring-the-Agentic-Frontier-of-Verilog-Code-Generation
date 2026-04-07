# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example of a custom model factory implementation with GPT, Claude, Gemini, and
third-party OpenRouter model support.
To use this factory:
1. For GPT, Claude, Kimi, MiniMax, Qwen: Set OPENROUTER_API_KEY in your .env file
2. For Gemini: Set GEMINI_API_KEY in your .env file
3. Run benchmark with the --custom-factory flag:
   python run_benchmark.py -f input.json -l -m openai/gpt-4o -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m anthropic/claude-sonnet-4-5 -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m gemini-2.5-pro -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m gemini-3-pro-preview -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m moonshotai/kimi-k2.5 -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m minimax/minimax-m2.5 -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m qwen/qwen3-max-thinking -c custom_model_factory.py
"""
import logging
import os
import sys
from typing import Optional, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.llm_lib.model_factory import ModelFactory
from src.config_manager import config
from openrouter_models import GPT_Instance, Claude_Instance, Gemini_Instance, OpenRouter_Instance
from subjective_score_model import SubjectiveScoreModel_Instance


class CustomModelFactory(ModelFactory):
    """
    Custom model factory with GPT, Claude, Gemini, and third-party OpenRouter model support.
    """

    def __init__(self):
        super().__init__()

        # Register GPT models (routed via OpenRouter)
        self.model_types["openai/gpt-4o"] = self._create_gpt_instance
        self.model_types["openai/gpt-4o-mini"] = self._create_gpt_instance
        self.model_types["openai/gpt-4.1"] = self._create_gpt_instance
        self.model_types["openai/gpt-4.1-mini"] = self._create_gpt_instance
        self.model_types["openai/o3"] = self._create_gpt_instance
        self.model_types["openai/o4-mini"] = self._create_gpt_instance

        # Register Claude models (routed via OpenRouter)
        self.model_types["anthropic/claude-sonnet-4-5"] = self._create_claude_instance
        self.model_types["anthropic/claude-opus-4-5"] = self._create_claude_instance
        self.model_types["anthropic/claude-3-5-haiku"] = self._create_claude_instance

        # Register Gemini models (direct Google SDK)
        self.model_types["gemini-2.5-pro"] = self._create_gemini_instance
        self.model_types["gemini-3.1-pro-preview"] = self._create_gemini_instance
        self.model_types["gemini-2.5-flash"] = self._create_gemini_instance

        # Register third-party OpenRouter models
        # kimi-k2.5 and minimax-m2.5: support reasoning effort parameter
        # qwen3-max-thinking: thinking always-on, no reasoning parameter needed
        self.model_types["moonshotai/kimi-k2.5"] = self._create_openrouter_instance
        self.model_types["minimax/minimax-m2.5"] = self._create_openrouter_instance
        self.model_types["qwen/qwen3-max-thinking"] = self._create_openrouter_instance
        self.model_types["z-ai/glm-5"] = self._create_openrouter_instance
        self.model_types["z-ai/glm-4.7"] = self._create_openrouter_instance

        # Register subjective scoring model
        self.model_types["sbj_score"] = self._create_sbj_score_instance

        logging.info(
            "Custom model factory initialized with GPT, Claude, Gemini, "
            "Kimi, MiniMax, Qwen, and subjective scoring support"
        )

    def _create_gpt_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a GPT model instance via OpenRouter."""
        return GPT_Instance(context=context, key=key, model=model_name)

    def _create_claude_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a Claude model instance via OpenRouter."""
        return Claude_Instance(context=context, key=key, model=model_name)

    def _create_gemini_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a Gemini model instance (e.g. gemini-2.5-pro, gemini-3-pro-preview)."""
        return Gemini_Instance(context=context, key=key, model=model_name)

    def _create_openrouter_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a generic OpenRouter model instance for third-party models."""
        return OpenRouter_Instance(context=context, key=key, model=model_name)

    def _create_sbj_score_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a subjective scoring model instance using a fixed Gemini judge model."""
        return SubjectiveScoreModel_Instance(context=context, key=key, model="gemini-3.1-pro-preview")


if __name__ == "__main__":
    factory = CustomModelFactory()

    # Test GPT via OpenRouter
    try:
        gpt_model = factory.create_model(model_name="openai/gpt-4o", context="You are a helpful assistant.")
        print(f"Successfully created GPT model: {gpt_model.model}")
        response = gpt_model.prompt(
            prompt="Generate a simple hello world program in SystemVerilog",
            files=["hello.sv"],
            timeout=60,
            category=9
        )
        print(f"GPT response: {response}")
    except Exception as e:
        print(f"GPT error: {e}")

    # Test Claude via OpenRouter
    try:
        claude_model = factory.create_model(model_name="anthropic/claude-sonnet-4-5", context="You are a helpful assistant.")
        print(f"Successfully created Claude model: {claude_model.model}")
        response = claude_model.prompt(
            prompt="Generate a simple hello world program in SystemVerilog",
            files=["hello.sv"],
            timeout=60,
            category=9
        )
        print(f"Claude response: {response}")
    except Exception as e:
        print(f"Claude error: {e}")

    # Test Kimi via OpenRouter
    try:
        kimi_model = factory.create_model(model_name="moonshotai/kimi-k2.5", context="You are a helpful assistant.")
        print(f"Successfully created Kimi model: {kimi_model.model}")
        response = kimi_model.prompt(
            prompt="Generate a simple hello world program in SystemVerilog",
            files=["hello.sv"],
            timeout=60,
            category=9
        )
        print(f"Kimi response: {response}")
    except Exception as e:
        print(f"Kimi error: {e}")