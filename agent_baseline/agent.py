#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple CVDP agent implementation for the agentic workflow.
This agent reads prompt.json and makes changes to files in the mounted directories.

Supports:
  - Google Gemini  (via google-genai SDK)        → set BACKEND=gemini
  - OpenRouter     (OpenAI-compat, any model)    → set BACKEND=openrouter
      covers Claude-via-OpenRouter, GPT-4o, etc.

Required env vars:
  BACKEND            : "gemini" | "openrouter"
  GEMINI_API_KEY     : required when BACKEND=gemini
  OPENROUTER_API_KEY : required when BACKEND=openrouter
  MODEL_NAME         : model string, e.g.
                         "gemini-2.0-flash"                  (gemini)
                         "anthropic/claude-sonnet-4-5"       (openrouter)
                         "openai/gpt-4o"                     (openrouter)
"""

import os
import json
import sys
import subprocess

# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a language model that has the following file operations available at your disposal:\n"
    "  - **List files in a directory** by running one of the following commands: \n"
    "    - `ls`\n    - `tree`\n"
    "  - **Read files** by using:\n    - `cat <filename>`\n"
    "  - **Write files** by using:\n    - `echo <content> > <filename>`\n"
    "  - **Compile Verilog** by using `iverilog` such as:\n"
    "    - `iverilog -o <output_filename>.out -g2012 <verilog_code_file> <verilog_testbench_file>`\n"
    "  - **Run Simulation** by using:\n    - `vvp <output_filename>.out`\n"
    "  - **Find current working directory** by using:\n    - `pwd`\n"
    "  - **Update the contents of a text file from a old content to new content**\n"
    "    - `sed -i  \"problematic_line_number s/problematic_statement/non_problematic_statement/\" Buggy_RTL_code.sv`\n"
    "  - **To access a specific line of the file**\n"
    "     - `awk 'NR==line_number' file_name.sv`\n\n"
    "You will be given a prompt and your task is to understand it and solve the given issue by using the "
    "above-mentioned commands as needed. In the final step, you should create a Linux patch to highlight "
    "the necessary file updates to achieve the targeted goal.\n\n"
    "  You will solve the problem step by step using the following approach of \n"
    "  - thought (thinking process of the step you're going to take)\n"
    "  - action (the command you will be running)\n"
    "  - observation (the output from the action)\n\n"
    "  The last step will be the final output summary and the patch itself in the following format \n"
    "  - thought (the summary of what you did and some introduction of the patch file itself)\n"
    "  - patch (a Linux-based patch that needs to be applied to reach the relevant solution)\n\n"
)

# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

def execute_shell(command: str, working_dir: str = "/code") -> str:
    """Execute a shell command and return combined stdout + stderr."""
    print(f"[TOOL] shell_exec: {command!r}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=60,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] Command timed out after 60 seconds."
    except Exception as e:
        return f"[error] {e}"

# Add more tool functions here — just register them in TOOL_DISPATCH below
TOOL_DISPATCH = {
    "shell_exec": lambda args: execute_shell(
        args["command"], args.get("working_dir", "/code")
    ),
}

# ---------------------------------------------------------------------------
# Tool schema: OpenAI / OpenRouter format
# ---------------------------------------------------------------------------

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": (
                "Execute a shell command on the host system and return its output. "
                "Use this to run ls, cat, echo, iverilog, vvp, sed, awk, pwd, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for the command (default: /code).",
                    },
                },
                "required": ["command"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Tool schema: Google Gemini format
# ---------------------------------------------------------------------------

GEMINI_TOOL_DECLARATIONS = [
    {
        "name": "shell_exec",
        "description": (
            "Execute a shell command on the host system and return its output. "
            "Use this to run ls, cat, echo, iverilog, vvp, sed, awk, pwd, etc."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "command": {
                    "type": "STRING",
                    "description": "The shell command to execute.",
                },
                "working_dir": {
                    "type": "STRING",
                    "description": "Working directory for the command (default: /code).",
                },
            },
            "required": ["command"],
        },
    }
]

# ---------------------------------------------------------------------------
# File helpers (unchanged from original)
# ---------------------------------------------------------------------------

def read_prompt() -> str:
    try:
        with open("/code/prompt.json", "r") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt", "")
    except Exception as e:
        print(f"Error reading prompt.json: {e}")
        return ""

def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def write_file(file_path: str, content: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

# ---------------------------------------------------------------------------
# Backend: OpenRouter  (OpenAI-compatible — same code works for Claude,
#                       GPT-4o, Mistral, etc. on OpenRouter)
# ---------------------------------------------------------------------------

def run_openrouter(prompt: str) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not found.  pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("MODEL_NAME", "anthropic/claude-sonnet-4-6")
    if not api_key:
        print("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": prompt},
    ]
    print(f"[OpenRouter] model={model}")

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # Replace the simple messages.append(msg) with this:
        messages.append(msg)

        # For thinking/reasoning models (minimax, qwen3-*-thinking, etc.),
        # the SDK surfaces reasoning as msg.reasoning_content or inside content blocks.
        # OpenRouter passes it back via a separate field — preserve it so the model
        # doesn't lose its chain of thought across turns.
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            # Already included in the message object the SDK built, nothing extra needed.
            # But log it so you can see the thinking:
            print(f"[THINKING] {msg.reasoning_content[:200]}...")

        # In the finish condition, also check if content is empty but reasoning isn't
        if finish_reason == "stop" or not msg.tool_calls:
            final_text = msg.content or getattr(msg, 'reasoning_content', None) or "(no text content)"
            print("\n=== Agent Final Response ===")
            print(final_text)
            break

        # Execute every requested tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            tool_result = TOOL_DISPATCH.get(fn_name, lambda _: f"[error] unknown tool {fn_name}")(fn_args)
            print(f"[TOOL RESULT] {tool_result[:400]}{'...' if len(tool_result) > 400 else ''}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

# ---------------------------------------------------------------------------
# Backend: Google Gemini  (google-generativeai SDK)
# ---------------------------------------------------------------------------

def run_gemini(prompt: str) -> None:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("google-genai package not found.  pip install google-genai")
        sys.exit(1)

    api_key    = os.environ.get("GEMINI_API_KEY")
    model_name = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
    if not api_key:
        print("GEMINI_API_KEY not set.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print(f"[Gemini] model={model_name}")

    # Build the tool declaration using the new SDK's types
    shell_exec_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="shell_exec",
                description=(
                    "Execute a shell command on the host system and return its output. "
                    "Use this to run ls, cat, echo, iverilog, vvp, sed, awk, pwd, etc."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "command": types.Schema(
                            type=types.Type.STRING,
                            description="The shell command to execute.",
                        ),
                        "working_dir": types.Schema(
                            type=types.Type.STRING,
                            description="Working directory for the command (default: /code).",
                        ),
                    },
                    required=["command"],
                ),
            )
        ]
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_MESSAGE,
        tools=[shell_exec_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Conversation history — new SDK uses a list of Content objects
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    while True:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        candidate   = response.candidates[0]
        parts       = candidate.content.parts
        text_parts  = [p.text for p in parts if p.text]
        fn_calls    = [p.function_call for p in parts if p.function_call]

        if text_parts:
            print("[Model]", " ".join(text_parts))

        # Append model turn to history
        contents.append(types.Content(role="model", parts=parts))

        if not fn_calls:
            print("\n=== Agent Final Response ===")
            print(" ".join(text_parts) if text_parts else "(no text content)")
            break

        # Execute tool calls and build a single user turn with all results
        result_parts = []
        for fc in fn_calls:
            fn_args     = dict(fc.args)
            tool_result = TOOL_DISPATCH.get(
                fc.name, lambda _: f"[error] unknown tool {fc.name}"
            )(fn_args)
            print(f"[TOOL RESULT] {tool_result[:400]}{'...' if len(tool_result) > 400 else ''}")
            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": tool_result},
                    )
                )
            )

        # Feed all results back as a single user turn
        contents.append(types.Content(role="user", parts=result_parts))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Starting CVDP agent...")
    backend = os.environ.get("BACKEND", "openrouter").lower()
    prompt  = read_prompt()
    if not prompt:
        print("No prompt found in /code/prompt.json — exiting.")
        sys.exit(1)

    print(f"Prompt loaded ({len(prompt)} chars). Backend: {backend}")

    if backend == "gemini":
        run_gemini(prompt)
    elif backend == "openrouter":
        run_openrouter(prompt)
    else:
        print(f"Unknown BACKEND={backend!r}. Set to 'gemini' or 'openrouter'.")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()