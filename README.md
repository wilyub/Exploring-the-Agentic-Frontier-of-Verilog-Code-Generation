# Exploring the Agentic Frontier of Verilog Code Generation

This is a respository for exploring the Verilog code generation benchmark "Comprehensive Verilog Design Problems" (CVDP) with agentic modeling. This repository contains the code necessary to run CVDP in various agentic settings as outlined in our paper: [Link](https://arxiv.org/abs/2603.19347).

## Setup
1. Follow the quick start setps found on the CVDP github: [Link](https://github.com/NVlabs/cvdp_benchmark?tab=readme-ov-file#quick-start)
2. Download the CVDP datasets from huggingface into this directory. Our work focuses on the "agent code generation no commercial" subset of the dataset. [Link](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset)
3. Build the agents you would like to use by going to their directory and calling the following command:
```
./build_agent.sh --clean
```
4. Run the agentic version of the benchmark using the following command:
```
./run_samples.py -f cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl -l -g cvdp-example-agent -p your/desired/directory -n 5 -k 1 --agent-backend openrouter --agent-model gemini-3.1-pro-preview
```
Compatible models can be found in custom_model_factory_router.py.
5. Run the non-agentic version of the benchmark using the following command:
```
./run_samples.py -f cvdp_v1.0.2_agentic_code_generation_no_commercial.jsonl -l -m gemini-3.1-pro-preview -p your/desired/directory -n 5 -k 1 --force-copilot -c custom_model_factory_router.py
```