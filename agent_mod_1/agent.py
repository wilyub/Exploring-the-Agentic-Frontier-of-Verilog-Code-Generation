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
    "You are a Verilog hardware design assistant. Your task is to analyze, debug, or generate "
    "Verilog/SystemVerilog code based on a given prompt.\n\n"
    "You MUST follow this exact sequence of steps — do not skip or reorder them:\n\n"

    "## STEP 1 — Discover and read all files\n"
    "Run `ls -R` to list every file in the working directory. Then use `cat <filename>` to read "
    "EVERY file you find (source files, testbenches, specs, READMEs, etc.). Do not proceed until "
    "you have read all files in full.\n\n"

    "## STEP 2 — Plan your changes\n"
    "Think carefully about what edits or new files are required to satisfy the prompt. "
    "Write out your plan explicitly before touching any file. "
    "IMPORTANT: stay strictly within what the prompt and the files specify. "
    "Do NOT infer extra requirements, add unrequested features, or change anything not directly "
    "called for by the prompt or the existing specifications.\n\n"

    "## STEP 3 — Apply changes\n"
    "Implement your plan from Step 2 by modifying or creating files with Linux commands "
    "(`sed`, `echo`, `awk`, `tee`, `cp`, `mv`, etc.). "
    "Make only the changes you planned in Step 2.\n\n"

    "## STEP 4 — Compile and iterate\n"
    "Compile the modified Verilog using the `iverilog_compile` tool. "
    "If compilation fails, read the error message carefully, return to Step 2, revise your plan, "
    "and repeat Steps 3–4 until compilation succeeds. "
    "Do NOT move to Step 5 if there are any compilation errors.\n\n"

    "## STEP 5 — Signal completion\n"
    "Once compilation succeeds and you are confident the solution is correct, call the "
    "`task_complete` tool with a brief summary of what you did. "
    "Do not call `task_complete` before a successful compile.\n\n"

    "---\n"
    "At each step, structure your reasoning as:\n"
    "  - thought   : what you are about to do and why\n"
    "  - action    : the tool call / command\n"
    "  - observation : the result\n\n"
    "The final step should include:\n"
    "  - thought : summary of all changes made\n"
    "  - patch   : a Linux unified diff (patch) capturing every file change\n"
)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def execute_shell(command: str, working_dir: str = "/code") -> str:
    """Execute a general-purpose shell command and return combined stdout + stderr.
    Intended for filesystem/text operations only (ls, cat, echo, sed, awk, etc.).
    Do NOT use this for iverilog or vvp — use the dedicated tools instead.
    """
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


def iverilog_compile(
    source_files: list,
    output_name: str = "sim",
    working_dir: str = "/code",
    extra_flags: str = "",
) -> str:
    """Compile Verilog/SystemVerilog with iverilog.
    Returns 'Compilation successful.' or the error output only (no stdout noise).
    """
    sources = " ".join(source_files)
    cmd = f"iverilog -o {output_name}.out -g2012 {extra_flags} {sources} 2>&1"
    print(f"[TOOL] iverilog_compile: {cmd!r}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,   # combined via 2>&1 in cmd
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=working_dir,
            timeout=120,
        )
        output = result.stdout.strip()
        if result.returncode == 0:
            return "Compilation successful."
        # Return only error lines to avoid token bloat
        error_lines = [l for l in output.splitlines() if l.strip()]
        return "Compilation failed:\n" + "\n".join(error_lines)
    except subprocess.TimeoutExpired:
        return "[error] iverilog timed out after 120 seconds."
    except Exception as e:
        return f"[error] {e}"


def vvp_simulate(
    output_name: str = "sim",
    working_dir: str = "/code",
    timeout_seconds: int = 30,
) -> str:
    """Run a compiled VVP simulation.
    Returns 'Simulation passed.' if the process exits 0 with no $fatal/$error lines,
    otherwise returns only the relevant failure lines (not full waveform dumps).
    """
    cmd = f"vvp {output_name}.out"
    print(f"[TOOL] vvp_simulate: {cmd!r}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=timeout_seconds,
        )
        combined = (result.stdout + "\n" + result.stderr).strip()
        lines = combined.splitlines()

        # Detect explicit failure signals
        FAIL_KEYWORDS = ("$fatal", "$error", "FAILED", "ERROR", "ASSERTION", "mismatch")
        failure_lines = [
            l for l in lines
            if any(kw.lower() in l.lower() for kw in FAIL_KEYWORDS)
        ]

        if result.returncode != 0 or failure_lines:
            snippet = "\n".join(failure_lines) if failure_lines else "\n".join(lines[-20:])
            return f"Simulation failed:\n{snippet}"

        return "Simulation passed."
    except subprocess.TimeoutExpired:
        return f"[error] vvp simulation timed out after {timeout_seconds} seconds."
    except Exception as e:
        return f"[error] {e}"


def task_complete(summary: str) -> str:
    """Signal that the agent has finished successfully."""
    print(f"[TOOL] task_complete called: {summary!r}")
    return "__TASK_COMPLETE__"


TOOL_DISPATCH = {
    "shell_exec":       lambda args: execute_shell(
                            args["command"], args.get("working_dir", "/code")
                        ),
    "iverilog_compile": lambda args: iverilog_compile(
                            args["source_files"],
                            args.get("output_name", "sim"),
                            args.get("working_dir", "/code"),
                            args.get("extra_flags", ""),
                        ),
    "vvp_simulate":     lambda args: vvp_simulate(
                            args.get("output_name", "sim"),
                            args.get("working_dir", "/code"),
                            args.get("timeout_seconds", 30),
                        ),
    "task_complete":    lambda args: task_complete(args.get("summary", "")),
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
                "Execute a general-purpose shell command (ls, cat, echo, sed, awk, pwd, tee, cp, mv, etc.). "
                "For filesystem and text manipulation ONLY. "
                "Do NOT use this for iverilog or vvp — use iverilog_compile and vvp_simulate instead."
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
    },
    {
        "type": "function",
        "function": {
            "name": "iverilog_compile",
            "description": (
                "Compile Verilog/SystemVerilog source files using iverilog. "
                "Returns 'Compilation successful.' or a concise error message. "
                "Always use this instead of running iverilog via shell_exec."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Verilog/SystemVerilog source files to compile (in order).",
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Base name for the compiled output file (default: 'sim').",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: /code).",
                    },
                    "extra_flags": {
                        "type": "string",
                        "description": "Any additional iverilog flags (e.g. '-I include/').",
                    },
                },
                "required": ["source_files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vvp_simulate",
            "description": (
                "Run a compiled VVP simulation. "
                "Returns 'Simulation passed.' on success, or a filtered snippet of failure lines. "
                "Always use this instead of running vvp via shell_exec."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_name": {
                        "type": "string",
                        "description": "Base name of the compiled .out file to run (default: 'sim').",
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory (default: /code).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Max seconds to allow the simulation to run (default: 30).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Signal that the task is fully complete. Call this ONLY after a successful compilation. "
                "Include a brief summary of every change made and the final patch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Human-readable summary of what was done, followed by the unified diff patch.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool schema: Google Gemini format
# ---------------------------------------------------------------------------

def _build_gemini_tools():
    from google.genai import types
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="shell_exec",
                description=(
                    "Execute a general-purpose shell command (ls, cat, echo, sed, awk, pwd, tee, cp, mv, etc.). "
                    "For filesystem and text manipulation ONLY. "
                    "Do NOT use this for iverilog or vvp — use iverilog_compile and vvp_simulate instead."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "command": types.Schema(type=types.Type.STRING, description="The shell command to execute."),
                        "working_dir": types.Schema(type=types.Type.STRING, description="Working directory (default: /code)."),
                    },
                    required=["command"],
                ),
            ),
            types.FunctionDeclaration(
                name="iverilog_compile",
                description=(
                    "Compile Verilog/SystemVerilog source files using iverilog. "
                    "Returns 'Compilation successful.' or a concise error message. "
                    "Always use this instead of running iverilog via shell_exec."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "source_files": types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(type=types.Type.STRING),
                            description="List of Verilog/SystemVerilog source files to compile (in order).",
                        ),
                        "output_name": types.Schema(type=types.Type.STRING, description="Base name for output (default: 'sim')."),
                        "working_dir": types.Schema(type=types.Type.STRING, description="Working directory (default: /code)."),
                        "extra_flags": types.Schema(type=types.Type.STRING, description="Additional iverilog flags."),
                    },
                    required=["source_files"],
                ),
            ),
            types.FunctionDeclaration(
                name="vvp_simulate",
                description=(
                    "Run a compiled VVP simulation. "
                    "Returns 'Simulation passed.' on success, or a filtered snippet of failure lines. "
                    "Always use this instead of running vvp via shell_exec."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "output_name": types.Schema(type=types.Type.STRING, description="Base name of the compiled .out file (default: 'sim')."),
                        "working_dir": types.Schema(type=types.Type.STRING, description="Working directory (default: /code)."),
                        "timeout_seconds": types.Schema(type=types.Type.INTEGER, description="Max seconds for simulation (default: 30)."),
                    },
                    required=[],
                ),
            ),
            types.FunctionDeclaration(
                name="task_complete",
                description=(
                    "Signal that the task is fully complete. Call this ONLY after a successful compilation. "
                    "Include a brief summary of every change made and the final patch."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "summary": types.Schema(
                            type=types.Type.STRING,
                            description="Human-readable summary of what was done, followed by the unified diff patch.",
                        ),
                    },
                    required=["summary"],
                ),
            ),
        ]
    )

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_prompt() -> str:
    try:
        with open("/code/prompt.json", "r") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt", "")
    except Exception as e:
        print(f"Error reading prompt.json: {e}")
        return ""

# ---------------------------------------------------------------------------
# Backend: OpenRouter  (OpenAI-compatible)
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

        messages.append(msg)

        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            print(f"[THINKING] {msg.reasoning_content[:200]}...")

        # No more tool calls → natural stop
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

            # Check for task_complete signal before appending
            if tool_result == "__TASK_COMPLETE__":
                print("\n=== Task Complete — agent signalled successful finish ===")
                sys.exit(0)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

# ---------------------------------------------------------------------------
# Backend: Google Gemini
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

    shell_exec_tool = _build_gemini_tools()

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_MESSAGE,
        tools=[shell_exec_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

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

        contents.append(types.Content(role="model", parts=parts))

        if not fn_calls:
            print("\n=== Agent Final Response ===")
            print(" ".join(text_parts) if text_parts else "(no text content)")
            break

        result_parts = []
        for fc in fn_calls:
            fn_args     = dict(fc.args)
            tool_result = TOOL_DISPATCH.get(
                fc.name, lambda _: f"[error] unknown tool {fc.name}"
            )(fn_args)
            print(f"[TOOL RESULT] {tool_result[:400]}{'...' if len(tool_result) > 400 else ''}")

            # Check for task_complete signal
            if tool_result == "__TASK_COMPLETE__":
                print("\n=== Task Complete — agent signalled successful finish ===")
                sys.exit(0)

            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": tool_result},
                    )
                )
            )

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