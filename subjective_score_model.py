# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Subjective scoring model implementation for the CVDP benchmark system.

This model is used to evaluate how well a model's response matches
a reference answer for subjective evaluation tasks.

It can be created from a ModelFactory instance using:
factory.create_model("sbj_score")
"""

import os
import logging
import json
import re
from typing import Optional, Any, Dict, List
import openai
from src.config_manager import config

try:
    import google.genai as genai
except ImportError:
    genai = None

logging.basicConfig(level=logging.INFO)

class SubjectiveScoreModel_Instance:
    """
    Implementation of a subjective scoring model.

    This model is used to calculate subjective scores by comparing responses to reference answers.
    Supports OpenAI-compatible models and Google Gemini models (detected by "gemini-" prefix).
    """

    def __init__(self, context: Any = None, key: Optional[str] = None, model: str = None):
        """
        Initialize a subjective scoring model instance.

        Args:
            context: Not used for scoring models
            key: API key (OpenAI key for non-Gemini models; ignored for Gemini, uses GEMINI_API_KEY)
            model: The model version to use
        """
        if model is None:
            model = config.get("DEFAULT_MODEL")
        self.model = model
        self.debug = False

        if self.model.startswith("gemini-"):
            # Use Google Gemini SDK
            if genai is None:
                raise ImportError(
                    "The 'google-genai' package is required for Gemini scoring. "
                    "Install it with 'pip install google-genai'."
                )
            gemini_api_key = config.get("GEMINI_API_KEY")
            if gemini_api_key is None:
                raise ValueError(
                    "Unable to create Gemini Subjective Scoring Model - No GEMINI_API_KEY provided"
                )
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            self.client = None
            logging.info(f"Created Gemini Subjective Scoring Model. Using model: {self.model}")
        else:
            # Use OpenAI-compatible client
            api_key = config.get("OPENAI_USER_KEY")
            if (key is None) and (api_key is None):
                raise ValueError("Unable to create Subjective Scoring Model - No API key provided")
            actual_key = key if key is not None else api_key
            self.client = openai.OpenAI(api_key=actual_key)
            self.gemini_client = None
            logging.info(f"Created Subjective Scoring Model using the provided key. Using model: {self.model}")
    
    def set_debug(self, debug: bool = True) -> None:
        """
        Enable or disable debug mode.
        
        Args:
            debug: Whether to enable debug mode (default: True)
        """
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")

    @property
    def requires_evaluation(self) -> bool:
        """
        Whether this model requires harness evaluation.
        
        Default is True for backward compatibility.
        
        Returns:
            bool: True (standard models require evaluation)
        """
        return True
    
    def subjective_score(self, response: str, reference: str, problem_prompt: str = "") -> float:
        """
        Calculate a subjective score for a response compared to a reference answer.

        Args:
            response: The model-generated response to evaluate
            reference: The reference (golden) answer to compare against
            problem_prompt: The original problem prompt for context

        Returns:
            A normalized score from 0.0-1.0 where 1.0 is perfect match and 0.0 is no match
        """
        system_prompt = """You are an expert at evaluating the quality of responses compared to reference solutions.
Your task is to score how well a candidate response matches the reference solution on a scale from 0.0 to 1.0,
where 0.0 means no match at all and 1.0 means a perfect match.

Important: You should evaluate the responses ONLY in relation to the original problem or question that was asked.
Focus on how well each response addresses the specific requirements and needs of the original problem prompt.

Look for the following aspects when scoring:
1. Relevance - how well does each response address the specific question or problem posed?
2. Semantic similarity - do the responses convey the same meaning in the context of the original problem?
3. Content completeness - does the candidate response include all the necessary information required by the prompt?
4. Correctness - is the candidate response accurate and correct with respect to what was asked?
5. Style and format consistency - does the candidate response follow the same style as the reference?

You must be critical and objective in your assessment. Provide a numeric score as a floating point number.

The response should be in the form of a JSON object with the following fields:
{
    "score": <float>,
    "reasoning": <string>
}
"""

        user_prompt = f"""Please evaluate the following candidate response against the reference solution.
Score the match on a scale from 0.0 to 1.0, where 0.0 means no match at all and 1.0 means a perfect match.

Original Problem/Question:
```
{problem_prompt}
```

Reference Solution:
```
{reference}
```

Candidate Response:
```
{response}
```

Important: Evaluate the candidate response ONLY on how well it addresses the original problem compared to the reference solution.
Ignore aspects that aren't relevant to the original problem.

Provide your score as a single number from 0.0 to 1.0.

An example response is:
{{
    "score": 0.85,
    "reasoning": "The candidate response addresses the original problem well, but lacks some depth in the analysis."
}}
"""

        if self.debug:
            print(f"Sending scoring request using model: {self.model}")

        try:
            if self.gemini_client is not None:
                # Gemini path: combine system + user prompt into a single content string
                full_prompt = system_prompt + "\n\n" + user_prompt
                gemini_response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=1024,
                        temperature=0.1,
                    ),
                )
                result_text = (getattr(gemini_response, "text", None) or "").strip()
            else:
                # OpenAI-compatible path
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                result_text = completion.choices[0].message.content

            # Parse JSON from the response
            try:
                if "```json" in result_text:
                    json_content = result_text.split("```json")[1].rsplit("```", 1)[0].strip()
                elif "```" in result_text:
                    json_content = result_text.split("```")[1].rsplit("```", 1)[0].strip()
                else:
                    json_content = result_text

                result = json.loads(json_content)

                if "score" in result and isinstance(result["score"], (int, float)):
                    score = float(result["score"])
                    score = max(0.0, min(1.0, score))
                    if self.debug:
                        print(f"Score: {score}/1.0")
                        print(f"Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                    return score
                else:
                    logging.error("Invalid JSON schema in response")
                    return 0.0

            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Failed to parse JSON: {str(e)}")
                return 0.0

        except Exception as e:
            logging.error(f"Error in subjective scoring: {str(e)}")
            return 0.0