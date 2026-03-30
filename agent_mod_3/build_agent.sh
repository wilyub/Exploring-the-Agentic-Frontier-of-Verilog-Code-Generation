#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pass --no-cache-base as argument to force rebuild of base image
# Usage:
#   ./build_agent.sh              → fast rebuild (agent only if base exists)
#   ./build_agent.sh --clean      → full rebuild from scratch (use when fixing base deps)

if [ "$1" = "--clean" ]; then
    echo "Clean build: rebuilding base from scratch (this takes ~15 minutes)..."
    docker build --no-cache --platform linux/amd64 -f Dockerfile-base -t cvdp-agent-mod-3-base .
else
    echo "Fast build: using cached base..."
    docker build --platform linux/amd64 -f Dockerfile-base -t cvdp-agent-mod-3-base .
fi

docker build --no-cache --platform linux/amd64 -f Dockerfile-agent -t cvdp-agent-mod-3 .