# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example of a custom model factory implementation with GPT and Gemini support.

To use this factory:
1. For GPT: Set OPENAI_API_KEY in your .env file
2. For Gemini: Set GEMINI_API_KEY in your .env file
3. Run benchmark with the --custom-factory flag:
   python run_benchmark.py -f input.json -l -m gpt-5.2 -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m gemini-2.5-pro -c custom_model_factory.py
   python run_benchmark.py -f input.json -l -m gemini-3-pro-preview -c custom_model_factory.py
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
from gpt_instance import GPT_Instance, Gemini_Instance
from examples.subjective_score_model import SubjectiveScoreModel_Instance


class CustomModelFactory(ModelFactory):
    """
    Custom model factory with GPT and Gemini support.
    """

    def __init__(self):
        super().__init__()

        # Register GPT models
        self.model_types["gpt-5.2"] = self._create_gpt_instance
        self.model_types["gpt-5-2025-08-07"] = self._create_gpt_instance
        self.model_types["gpt-4o"] = self._create_gpt_instance
        self.model_types["gpt-4o-mini"] = self._create_gpt_instance

        # Register Gemini models
        self.model_types["gemini-2.5-pro"] = self._create_gemini_instance
        self.model_types["gemini-3-pro-preview"] = self._create_gemini_instance

        # Register subjective scoring model
        self.model_types["sbj_score"] = self._create_sbj_score_instance

        logging.info(
            "Custom model factory initialized with GPT, Gemini, and subjective scoring support"
        )

    def _create_gpt_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a GPT model instance"""
        return GPT_Instance(context=context, key=key, model=model_name)

    def _create_gemini_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a Gemini model instance (e.g. gemini-2.5-pro, gemini-3-pro-preview)"""
        return Gemini_Instance(context=context, key=key, model=model_name)

    def _create_sbj_score_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a subjective scoring model instance"""
        return SubjectiveScoreModel_Instance(context=context, key=key, model=model_name)


if __name__ == "__main__":
    factory = CustomModelFactory()
    
    try:
        gpt_model = factory.create_model(model_name="gpt-5.2", context="You are a helpful assistant.")
        print(f"Successfully created GPT model: {gpt_model.model}")
        
        response = gpt_model.prompt(
            prompt="Generate a simple hello world program in SystemVerilog",
            files=["hello.sv"],
            timeout=60,
            category=9  # Q&A on RTL category
        )
        print(f"Model response: {response}")
    except Exception as e:
        print(f"Error: {e}")
