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
import tempfile

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

    "## STEP 4 — Verify your implementation\n"
    "Run all applicable verification tools in this order. Each tool targets different bug classes "
    "— use ALL of them, not just the first one that passes:\n\n"
    "  4a. `iverilog_compile`  — confirms the RTL is syntactically valid.\n"
    "  4b. `verilator_lint`    — catches semantic issues iverilog misses: blocking/non-blocking "
    "mixing, implicit nets, sensitivity list errors, width mismatches.\n"
    "  4c. `yosys_lint`        — catches structural issues: undriven outputs, multiply-driven nets, "
    "undeclared signals, port mismatches.\n"
    "  4d. `yosys_synth`       — catches synthesis-time issues: unintended latches from incomplete "
    "if/case, combinational loops, and gives a gate-count sanity check.\n"
    "  4e. `get_module_ports`  — confirms port names, directions, and widths match the spec.\n"
    "  4f. `formal_verify`     — if the design has assertions (`assert`/`assume`/`cover`), run "
    "bounded model checking. A formal PASS is the strongest correctness guarantee available.\n"
    "  4g. Self-test simulation — write and run a directed testbench (details below).\n\n"
    "CRITICAL VERIFICATION RULE: When reviewing tool output, you MUST distinguish between:\n"
    "  - Warnings/errors in files YOU created or modified → these must be fixed.\n"
    "  - Warnings/errors in pre-existing files that you did NOT touch → IGNORE these completely. "
    "Do NOT modify pre-existing files just to silence linter warnings unless the prompt explicitly "
    "asks you to fix those files. Touching pre-existing files to fix warnings they had before your "
    "changes is out of scope and risks breaking correct code.\n\n"
    "If ANY tool reports an error or unexpected warning IN YOUR CHANGED FILES ONLY, return to "
    "Step 2, revise your plan, re-apply changes in Step 3, and re-run verification. "
    "Repeat until all tools are clean on the files you own.\n\n"

    "### STEP 4g — Self-test simulation\n"
    "After steps 4a–4f pass, you MUST write and run a small directed testbench to verify "
    "functional correctness. This is mandatory — do not skip it.\n\n"

    "**Part 1 — Derive test vectors (do this in your thought, before writing any code):**\n"
    "Re-read the specification/prompt. Identify 3–5 concrete, non-trivial input scenarios "
    "that exercise the key behaviours (e.g. reset behaviour, boundary values, mode switches, "
    "known encoding/decoding pairs, arithmetic edge cases). For each scenario write down:\n"
    "  - The exact input values you will apply\n"
    "  - The exact output values you expect, calculated by hand from the spec\n"
    "Be precise with bit widths and radices (e.g. `8'hA3`, `4'd7`). "
    "Do not guess — if you cannot derive the expected output from the spec, pick a simpler "
    "test case that you can.\n\n"

    "**Part 2 — Write the testbench:**\n"
    "Create `/tmp/self_test.sv` using `shell_exec`. Requirements:\n"
    "  - `\\`timescale 1ns/1ps`, module name `self_test_tb`\n"
    "  - Declare an integer `fail_count = 0`\n"
    "  - Instantiate only the module(s) you created/modified\n"
    "  - For combinational logic: apply inputs, `#10`, then check outputs\n"
    "  - For clocked logic: generate a clock, apply reset, then drive inputs and check "
    "outputs after the appropriate number of rising edges\n"
    "  - After each check: "
    "`$display(\"TEST <n>: got=%h exp=%h %s\", actual, expected, (actual===expected) ? \"PASS\" : \"FAIL\");`\n"
    "  - On mismatch: `fail_count = fail_count + 1;`\n"
    "  - End of testbench:\n"
    "    `if (fail_count) begin $display(\"SELF-TEST FAILED: %0d/%0d\", fail_count, total); "
    "$finish(1); end`\n"
    "    `else begin $display(\"SELF-TEST PASSED: all %0d tests\", total); $finish(0); end`\n"
    "  - Keep it under 100 lines — this is a directed sanity check, not an exhaustive suite\n\n"

    "**Part 3 — Run and interpret:**\n"
    "  1. `iverilog_compile` with your RTL files + `/tmp/self_test.sv`\n"
    "  2. `vvp_simulate` — read every output line\n"
    "  3. If `SELF-TEST PASSED`: proceed to Step 5.\n"
    "  4. If any test prints FAIL: reason through why the actual output differs from expected. "
    "Two possibilities — (a) your RTL is wrong → fix it and restart from Step 3; "
    "(b) your expected value was wrong → re-derive from the spec carefully and confirm before "
    "accepting. Do NOT call `task_complete` if any self-test failed without a clear resolution.\n\n"

    "## STEP 5 — Signal completion\n"
    "Once all applicable verification tools pass and you are confident the solution is correct, "
    "call the `task_complete` tool with a brief summary of what you did. "
    "Do not call `task_complete` before a successful `iverilog_compile`.\n\n"

    "---\n"
    "At each step, structure your reasoning as:\n"
    "  - thought     : what you are about to do and why\n"
    "  - action      : the tool call / command\n"
    "  - observation : the result\n"
)

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _run(cmd: str, cwd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Helper: run a shell command, capturing stdout+stderr separately."""
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=timeout
    )


# Maximum characters returned by shell_exec to avoid context overflow.
_SHELL_OUTPUT_LIMIT = 30000

# Extensions that are binary/compiled and should never be cat'd.
_BINARY_EXTENSIONS = (".out", ".vcd", ".fst", ".vvp", ".o", ".a", ".so", ".bin", ".hex")

def execute_shell(command: str, working_dir: str = "/code") -> str:
    """Execute a general-purpose shell command (ls, cat, echo, sed, awk, etc.).
    Filesystem and text manipulation ONLY — do NOT use for iverilog/vvp/yosys/verilator.
    Output is capped at 8000 characters to prevent context overflow.
    """
    print(f"[TOOL] shell_exec: {command!r}")

    # Guard: block cat on binary/compiled files to prevent context flooding
    cmd_stripped = command.strip()
    if cmd_stripped.startswith("cat "):
        for ext in _BINARY_EXTENSIONS:
            if ext in cmd_stripped:
                return (f"[blocked] Refusing to cat a binary/compiled file ({ext}). "
                        f"These files are not human-readable and will overflow the context. "
                        f"Use the dedicated tools (iverilog_compile, vvp_simulate) instead.")

    try:
        r = _run(command, working_dir)
        output = r.stdout
        if r.stderr:
            output += "\n[stderr]\n" + r.stderr
        output = output.strip() or "(no output)"
        # Truncate to prevent context overflow on unexpectedly large outputs
        if len(output) > _SHELL_OUTPUT_LIMIT:
            output = output[:_SHELL_OUTPUT_LIMIT] + f"\n...[truncated — output exceeded {_SHELL_OUTPUT_LIMIT} chars]"
        return output
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
    """Compile Verilog/SystemVerilog with iverilog (-g2012).
    Returns 'Compilation successful.' or filtered error lines only.
    """
    sources = " ".join(source_files)
    cmd = f"iverilog -o {output_name}.out -g2012 {extra_flags} {sources}"
    print(f"[TOOL] iverilog_compile: {cmd!r}")
    try:
        r = _run(cmd, working_dir, timeout=120)
        combined = (r.stdout + "\n" + r.stderr).strip()
        if r.returncode == 0:
            return "Compilation successful."
        error_lines = [l for l in combined.splitlines() if l.strip()]
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
    Returns 'Simulation passed.' or filtered failure lines only (no waveform dumps).
    """
    cmd = f"vvp {output_name}.out"
    print(f"[TOOL] vvp_simulate: {cmd!r}")
    try:
        r = _run(cmd, working_dir, timeout=timeout_seconds)
        combined = (r.stdout + "\n" + r.stderr).strip()
        lines = combined.splitlines()
        FAIL_KEYWORDS = ("$fatal", "$error", "FAILED", "ERROR", "ASSERTION", "mismatch")
        failure_lines = [l for l in lines if any(kw.lower() in l.lower() for kw in FAIL_KEYWORDS)]
        if r.returncode != 0 or failure_lines:
            snippet = "\n".join(failure_lines) if failure_lines else "\n".join(lines[-20:])
            return f"Simulation failed:\n{snippet}"
        return "Simulation passed."
    except subprocess.TimeoutExpired:
        return f"[error] vvp timed out after {timeout_seconds} seconds."
    except Exception as e:
        return f"[error] {e}"


def verilator_lint(
    source_files: list,
    top_module: str = "",
    working_dir: str = "/code",
    extra_flags: str = "",
) -> str:
    """Run Verilator in lint-only mode (-Wall).
    Catches: blocking/non-blocking mixing, implicit nets, sensitivity list errors, width mismatches.
    Returns 'Lint clean.' or filtered %Warning/%Error lines only.
    """
    sources = " ".join(source_files)
    top_flag = f"--top-module {top_module}" if top_module else ""
    cmd = f"verilator --lint-only -Wall {top_flag} {extra_flags} {sources}"
    print(f"[TOOL] verilator_lint: {cmd!r}")
    try:
        r = _run(cmd, working_dir, timeout=60)
        combined = (r.stdout + "\n" + r.stderr).strip()
        lines = [l for l in combined.splitlines() if l.strip()]
        error_lines = [l for l in lines if "%Error" in l or "%Warning" in l]
        if r.returncode != 0 or error_lines:
            return "Verilator lint issues:\n" + "\n".join(error_lines or lines[-30:])
        return "Lint clean."
    except FileNotFoundError:
        return "[skip] verilator not found on this system."
    except subprocess.TimeoutExpired:
        return "[error] verilator timed out after 60 seconds."
    except Exception as e:
        return f"[error] {e}"


def yosys_lint(
    source_files: list,
    top_module: str = "",
    working_dir: str = "/code",
) -> str:
    """Run Yosys structural lint: read → hierarchy -check → check.
    Catches: undeclared signals, port width mismatches, multiply-driven wires, undriven outputs.
    Returns 'Yosys lint clean.' or filtered issue lines.
    """
    reads = "\n".join(f"read_verilog -sv {f}" for f in source_files)
    hier  = f"hierarchy -check {'-top ' + top_module if top_module else ''}"
    script = f"{reads}\n{hier}\ncheck\n"
    print(f"[TOOL] yosys_lint: files={source_files} top={top_module!r}")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ys", delete=False) as tf:
            tf.write(script)
            script_path = tf.name
        r = _run(f"yosys -q {script_path}", working_dir, timeout=120)
        os.unlink(script_path)
        combined = (r.stdout + "\n" + r.stderr).strip()
        lines = [l for l in combined.splitlines() if l.strip()]
        ISSUE_KW = ("error", "warning", "not found", "multiple", "undriven", "width mismatch")
        issue_lines = [l for l in lines if any(kw in l.lower() for kw in ISSUE_KW)]
        if r.returncode != 0 or issue_lines:
            return "Yosys lint issues:\n" + "\n".join(issue_lines or lines[-30:])
        return "Yosys lint clean."
    except FileNotFoundError:
        return "[skip] yosys not found on this system."
    except subprocess.TimeoutExpired:
        return "[error] yosys lint timed out after 120 seconds."
    except Exception as e:
        return f"[error] {e}"


def yosys_synth(
    source_files: list,
    top_module: str = "",
    working_dir: str = "/code",
    target: str = "synth",
) -> str:
    """Synthesize the design with Yosys to reveal:
      - Inferred latches (incomplete if/case without default/else)
      - Combinational loops
      - Multi-driven nets
    Also returns a compact gate-count summary for sanity-checking design size.
    Returns filtered warnings + cell stats only.
    """
    reads = "\n".join(f"read_verilog -sv {f}" for f in source_files)
    top_flag = f"-top {top_module}" if top_module else ""
    script = f"{reads}\n{target} {top_flag}\nstat\n"
    print(f"[TOOL] yosys_synth: files={source_files} top={top_module!r}")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ys", delete=False) as tf:
            tf.write(script)
            script_path = tf.name
        r = _run(f"yosys {script_path}", working_dir, timeout=180)
        os.unlink(script_path)
        combined = (r.stdout + "\n" + r.stderr).strip()
        lines = combined.splitlines()

        WARN_KW = ("warning", "error", "latch", "loop", "multiple driver")
        warn_lines = [l for l in lines if any(kw in l.lower() for kw in WARN_KW)]

        # Pull the compact stat block
        stat_lines = []
        in_stat = False
        for l in lines:
            if l.strip().startswith("===") or "Number of cells" in l or "Number of wires" in l:
                in_stat = True
            if in_stat:
                stat_lines.append(l)
            if in_stat and l.strip() == "":
                break

        if r.returncode != 0:
            err = [l for l in lines if l.strip()]
            return "Synthesis failed:\n" + "\n".join(err[-30:])

        report = []
        report.append("Synthesis warnings:\n" + "\n".join(warn_lines) if warn_lines else "No synthesis warnings.")
        if stat_lines:
            report.append("Gate-count summary:\n" + "\n".join(stat_lines[:20]))
        return "\n\n".join(report)
    except FileNotFoundError:
        return "[skip] yosys not found on this system."
    except subprocess.TimeoutExpired:
        return "[error] yosys synthesis timed out after 180 seconds."
    except Exception as e:
        return f"[error] {e}"


def get_module_ports(
    source_file: str,
    module_name: str = "",
    working_dir: str = "/code",
) -> str:
    """Extract port list (name, direction, width) from a Verilog module via Yosys JSON backend.
    Use this to verify your implementation's interface matches the spec before compiling.
    """
    top_flag = f"-top {module_name}" if module_name else ""
    script = (
        f"read_verilog -sv {source_file}\n"
        f"hierarchy {top_flag}\n"
        f"proc\n"  # elaborate always blocks so JSON backend can process the design
        f"write_json /tmp/_ports_dump.json\n"
    )
    print(f"[TOOL] get_module_ports: file={source_file!r} module={module_name!r}")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ys", delete=False) as tf:
            tf.write(script)
            script_path = tf.name
        r = _run(f"yosys -q {script_path}", working_dir, timeout=60)
        os.unlink(script_path)
        if r.returncode != 0:
            return f"[error] yosys failed:\n{(r.stdout + r.stderr).strip()[-500:]}"

        with open("/tmp/_ports_dump.json") as jf:
            data = json.load(jf)

        modules = data.get("modules", {})
        target = module_name if module_name in modules else (list(modules.keys())[0] if modules else None)
        if not target:
            return "[error] No modules found in JSON dump."

        ports = modules[target].get("ports", {})
        rows = []
        for pname, pinfo in ports.items():
            direction = pinfo.get("direction", "?")
            bits      = pinfo.get("bits", [])
            width     = len(bits)
            rows.append(
                f"  {direction:6s}  [{width-1}:0]  {pname}" if width > 1
                else f"  {direction:6s}  [0:0]   {pname}"
            )
        return f"Module '{target}' ports:\n" + "\n".join(rows) if rows else f"Module '{target}' has no ports."
    except FileNotFoundError:
        return "[skip] yosys not found on this system."
    except subprocess.TimeoutExpired:
        return "[error] get_module_ports timed out."
    except Exception as e:
        return f"[error] {e}"


def formal_verify(
    source_files: list,
    top_module: str,
    working_dir: str = "/code",
    depth: int = 20,
) -> str:
    """Run bounded model checking via SymbiYosys (sby).
    Checks assert/assume/cover properties embedded in the RTL.
    PASS = no assertion fires within `depth` clock cycles (strongest guarantee available).
    Returns 'Formal PASS' or 'Formal FAIL' + relevant counterexample lines only.
    """
    print(f"[TOOL] formal_verify: files={source_files} top={top_module!r} depth={depth}")
    reads = "\n".join(f"  read -formal {f}" for f in source_files)
    sby_script = (
        "[options]\nmode bmc\ndepth {depth}\n\n"
        "[engines]\nsmtbmc\n\n"
        "[script]\n{reads}\nprep -top {top}\n\n"
        "[files]\n{files}\n"
    ).format(
        depth=depth,
        reads=reads,
        top=top_module,
        files="\n".join(source_files),
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sby", dir=working_dir, delete=False
        ) as tf:
            tf.write(sby_script)
            sby_path = tf.name

        r = _run(f"sby -f {sby_path}", working_dir, timeout=300)
        os.unlink(sby_path)
        combined = (r.stdout + "\n" + r.stderr).strip()
        lines = combined.splitlines()

        if "PASS" in combined and r.returncode == 0:
            return f"Formal PASS — no assertion violations within depth={depth}."

        FAIL_KW = ("FAIL", "Assert", "assert", "counterexample", "step", "ERROR")
        fail_lines = [l for l in lines if any(kw in l for kw in FAIL_KW)]
        snippet = "\n".join(fail_lines[:30]) if fail_lines else "\n".join(lines[-30:])
        return f"Formal FAIL (depth={depth}):\n{snippet}"
    except FileNotFoundError:
        return "[skip] sby (SymbiYosys) not found on this system."
    except subprocess.TimeoutExpired:
        return "[error] formal_verify timed out after 300 seconds."
    except Exception as e:
        return f"[error] {e}"


def task_complete(summary: str) -> str:
    """Signal that the agent has finished successfully."""
    print(f"[TOOL] task_complete called.")
    return "__TASK_COMPLETE__"


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

TOOL_DISPATCH = {
    "shell_exec":       lambda a: execute_shell(a["command"], a.get("working_dir", "/code")),
    "iverilog_compile": lambda a: iverilog_compile(
                            a["source_files"], a.get("output_name", "sim"),
                            a.get("working_dir", "/code"), a.get("extra_flags", "")),
    "vvp_simulate":     lambda a: vvp_simulate(
                            a.get("output_name", "sim"), a.get("working_dir", "/code"),
                            a.get("timeout_seconds", 30)),
    "verilator_lint":   lambda a: verilator_lint(
                            a["source_files"], a.get("top_module", ""),
                            a.get("working_dir", "/code"), a.get("extra_flags", "")),
    "yosys_lint":       lambda a: yosys_lint(
                            a["source_files"], a.get("top_module", ""),
                            a.get("working_dir", "/code")),
    "yosys_synth":      lambda a: yosys_synth(
                            a["source_files"], a.get("top_module", ""),
                            a.get("working_dir", "/code"), a.get("target", "synth")),
    "get_module_ports": lambda a: get_module_ports(
                            a["source_file"], a.get("module_name", ""),
                            a.get("working_dir", "/code")),
    "formal_verify":    lambda a: formal_verify(
                            a["source_files"], a["top_module"],
                            a.get("working_dir", "/code"), a.get("depth", 20)),
    "task_complete":    lambda a: task_complete(a.get("summary", "")),
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
                "Filesystem and text manipulation ONLY. "
                "Do NOT use for iverilog, vvp, yosys, or verilator — use their dedicated tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command":     {"type": "string", "description": "The shell command to execute."},
                    "working_dir": {"type": "string", "description": "Working directory (default: /code)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "iverilog_compile",
            "description": "Compile Verilog/SystemVerilog with iverilog -g2012. Returns 'Compilation successful.' or concise error lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_files": {"type": "array", "items": {"type": "string"}, "description": "Source files to compile, in order."},
                    "output_name":  {"type": "string", "description": "Base name for the .out file (default: 'sim')."},
                    "working_dir":  {"type": "string", "description": "Working directory (default: /code)."},
                    "extra_flags":  {"type": "string", "description": "Additional iverilog flags (e.g. '-I include/')."},
                },
                "required": ["source_files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vvp_simulate",
            "description": "Run a compiled VVP simulation. Returns 'Simulation passed.' or filtered failure lines — no waveform dumps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_name":     {"type": "string", "description": "Base name of the compiled .out file (default: 'sim')."},
                    "working_dir":     {"type": "string", "description": "Working directory (default: /code)."},
                    "timeout_seconds": {"type": "integer", "description": "Max seconds for the simulation (default: 30)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verilator_lint",
            "description": (
                "Run Verilator --lint-only -Wall. Catches: blocking/non-blocking mixing in always blocks, "
                "implicit net declarations, sensitivity list errors, port width mismatches. "
                "Returns 'Lint clean.' or filtered %Warning/%Error lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_files": {"type": "array", "items": {"type": "string"}, "description": "Source files to lint."},
                    "top_module":   {"type": "string", "description": "Top module name (optional but recommended)."},
                    "working_dir":  {"type": "string", "description": "Working directory (default: /code)."},
                    "extra_flags":  {"type": "string", "description": "Additional Verilator flags."},
                },
                "required": ["source_files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yosys_lint",
            "description": (
                "Run Yosys structural lint (read → hierarchy -check → check). "
                "Catches: undeclared signals, port width mismatches, multiply-driven wires, undriven outputs. "
                "Faster than full synthesis. Returns 'Yosys lint clean.' or issue lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_files": {"type": "array", "items": {"type": "string"}, "description": "Source files to check."},
                    "top_module":   {"type": "string", "description": "Top module name (optional)."},
                    "working_dir":  {"type": "string", "description": "Working directory (default: /code)."},
                },
                "required": ["source_files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yosys_synth",
            "description": (
                "Synthesize the design with Yosys. Reveals: inferred latches from incomplete if/case, "
                "combinational loops, multi-driven nets. Also returns a compact gate-count summary. "
                "Returns filtered warnings + cell stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_files": {"type": "array", "items": {"type": "string"}, "description": "Source files to synthesize."},
                    "top_module":   {"type": "string", "description": "Top module name (optional)."},
                    "working_dir":  {"type": "string", "description": "Working directory (default: /code)."},
                    "target":       {"type": "string", "description": "Yosys synth target (default: 'synth')."},
                },
                "required": ["source_files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_module_ports",
            "description": (
                "Extract the port list (name, direction, width) from a Verilog module via Yosys. "
                "Use this to verify your implementation's interface matches the spec before compiling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_file": {"type": "string", "description": "The Verilog source file containing the module."},
                    "module_name": {"type": "string", "description": "Module name to inspect (default: top module)."},
                    "working_dir": {"type": "string", "description": "Working directory (default: /code)."},
                },
                "required": ["source_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "formal_verify",
            "description": (
                "Run bounded model checking via SymbiYosys (sby) on assert/assume/cover properties. "
                "A PASS proves no assertion fires within `depth` clock cycles — the strongest correctness "
                "guarantee available. Returns 'Formal PASS' or 'Formal FAIL' + counterexample lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_files": {"type": "array", "items": {"type": "string"}, "description": "Source files (RTL + any property files)."},
                    "top_module":   {"type": "string", "description": "Top module for formal analysis."},
                    "working_dir":  {"type": "string", "description": "Working directory (default: /code)."},
                    "depth":        {"type": "integer", "description": "BMC depth in clock cycles (default: 20)."},
                },
                "required": ["source_files", "top_module"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": (
                "Signal that the task is fully complete. Call ONLY after iverilog_compile succeeds. "
                "Include a summary of every change made and the unified diff patch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of changes made, followed by the unified diff patch."},
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

    def _str(desc):
        return types.Schema(type=types.Type.STRING, description=desc)

    def _int(desc):
        return types.Schema(type=types.Type.INTEGER, description=desc)

    def _arr(desc):
        return types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description=desc)

    def _obj(props: dict, required: list = []):
        return types.Schema(type=types.Type.OBJECT, properties=props, required=required)

    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="shell_exec",
            description="General-purpose shell command (ls, cat, echo, sed, awk, etc.). NOT for iverilog/vvp/yosys/verilator.",
            parameters=_obj({"command": _str("Shell command."), "working_dir": _str("Working dir (default: /code).")}, ["command"]),
        ),
        types.FunctionDeclaration(
            name="iverilog_compile",
            description="Compile Verilog with iverilog -g2012. Returns 'Compilation successful.' or error lines.",
            parameters=_obj({
                "source_files": _arr("Source files to compile."),
                "output_name": _str("Output base name (default: sim)."),
                "working_dir": _str("Working dir (default: /code)."),
                "extra_flags": _str("Additional iverilog flags."),
            }, ["source_files"]),
        ),
        types.FunctionDeclaration(
            name="vvp_simulate",
            description="Run VVP simulation. Returns 'Simulation passed.' or filtered failure lines.",
            parameters=_obj({
                "output_name": _str("Compiled .out base name (default: sim)."),
                "working_dir": _str("Working dir (default: /code)."),
                "timeout_seconds": _int("Max seconds (default: 30)."),
            }),
        ),
        types.FunctionDeclaration(
            name="verilator_lint",
            description="Verilator --lint-only -Wall. Catches blocking/non-blocking mixing, implicit nets, width mismatches.",
            parameters=_obj({
                "source_files": _arr("Source files to lint."),
                "top_module": _str("Top module (optional)."),
                "working_dir": _str("Working dir (default: /code)."),
                "extra_flags": _str("Additional Verilator flags."),
            }, ["source_files"]),
        ),
        types.FunctionDeclaration(
            name="yosys_lint",
            description="Yosys structural lint. Catches undeclared signals, multiply-driven wires, port mismatches.",
            parameters=_obj({
                "source_files": _arr("Source files."),
                "top_module": _str("Top module (optional)."),
                "working_dir": _str("Working dir (default: /code)."),
            }, ["source_files"]),
        ),
        types.FunctionDeclaration(
            name="yosys_synth",
            description="Yosys synthesis. Reveals inferred latches, combinational loops. Returns warnings + gate-count.",
            parameters=_obj({
                "source_files": _arr("Source files."),
                "top_module": _str("Top module (optional)."),
                "working_dir": _str("Working dir (default: /code)."),
                "target": _str("Yosys synth target (default: synth)."),
            }, ["source_files"]),
        ),
        types.FunctionDeclaration(
            name="get_module_ports",
            description="Extract port list (name, direction, width) via Yosys. Verify interface matches spec.",
            parameters=_obj({
                "source_file": _str("Verilog source file."),
                "module_name": _str("Module to inspect (default: top)."),
                "working_dir": _str("Working dir (default: /code)."),
            }, ["source_file"]),
        ),
        types.FunctionDeclaration(
            name="formal_verify",
            description="SymbiYosys BMC on assert/assume/cover. PASS = no assertion fires within depth cycles.",
            parameters=_obj({
                "source_files": _arr("Source files (RTL + properties)."),
                "top_module": _str("Top module for formal analysis."),
                "working_dir": _str("Working dir (default: /code)."),
                "depth": _int("BMC depth in cycles (default: 20)."),
            }, ["source_files", "top_module"]),
        ),
        types.FunctionDeclaration(
            name="task_complete",
            description="Signal task is done. Call ONLY after successful compile. Include summary + patch.",
            parameters=_obj({"summary": _str("Summary of changes + unified diff patch.")}, ["summary"]),
        ),
    ])

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_prompt() -> str:
    try:
        with open("/code/prompt.json", "r") as f:
            return json.load(f).get("prompt", "")
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
    model   = os.environ.get("MODEL_NAME", "anthropic/claude-sonnet-4-6")
    if not api_key:
        print("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    client   = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": prompt},
    ]
    print(f"[OpenRouter] model={model}")

    while True:
        response = client.chat.completions.create(
            model=model, messages=messages, tools=OPENAI_TOOLS, tool_choice="auto",
        )
        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        messages.append(msg)

        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            print(f"[THINKING] {msg.reasoning_content[:200]}...")

        if finish_reason == "stop" or not msg.tool_calls:
            final_text = msg.content or getattr(msg, "reasoning_content", None) or "(no text content)"
            print("\n=== Agent Final Response ===")
            print(final_text)
            break

        for tc in msg.tool_calls:
            fn_name     = tc.function.name
            fn_args     = json.loads(tc.function.arguments)
            tool_result = TOOL_DISPATCH.get(fn_name, lambda _: f"[error] unknown tool {fn_name}")(fn_args)
            print(f"[TOOL RESULT] {tool_result[:400]}{'...' if len(tool_result) > 400 else ''}")

            if tool_result == "__TASK_COMPLETE__":
                print("\n=== Task Complete — agent signalled successful finish ===")
                sys.exit(0)

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})

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

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_MESSAGE,
        tools=[_build_gemini_tools()],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    while True:
        response   = client.models.generate_content(model=model_name, contents=contents, config=config)
        candidate  = response.candidates[0]
        parts      = candidate.content.parts
        text_parts = [p.text for p in parts if p.text]
        fn_calls   = [p.function_call for p in parts if p.function_call]

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
            tool_result = TOOL_DISPATCH.get(fc.name, lambda _: f"[error] unknown tool {fc.name}")(fn_args)
            print(f"[TOOL RESULT] {tool_result[:400]}{'...' if len(tool_result) > 400 else ''}")

            if tool_result == "__TASK_COMPLETE__":
                print("\n=== Task Complete — agent signalled successful finish ===")
                sys.exit(0)

            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name, response={"result": tool_result}
                )
            ))

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