# vLLM-Omni Distributed Testing for Wan2.2

This directory contains standalone testing scripts to run highly-optimized distributed inference for Wan2.2 video generation on bare-metal GPU instances (like RunPod/Vast.ai).

Before modifying your main `cog` Docker deployment, these scripts allow you to verify that **vLLM-Omni** runs natively on your GPU hardware and tests its Tensor Parallelism speedups.

### 1. Setup Environment
To install the `vllm-omni` engine, upgrade the required PyTorch dependencies, and automatically download the HuggingFace `Diffusers` format of the Wan2.2 weights (required by vLLM-Omni), run the setup bash script from your terminal:
```bash
bash vllm_setup.sh
```

### 2. Sample Run Commands (`generate_vllm.py`)

The `generate_vllm.py` script acts as the optimized equivalent of the original `generate.py`. It explicitly invokes the `vllm_omni.entrypoints.omni.Omni` Python class, mathematically distributing the immense 1.1 million generation tokens across your GPUs.

#### Single GPU Inference (e.g., 1x L40 or 1x A100)
```bash
python3 generate_vllm.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "A cinematic video of a cyberpunk city at night with neon lights reflecting in puddles"
```

#### Distributed Inference (2x GPUs)
By simply passing `--tensor_parallel_size 2`, vLLM will automatically divide the 56GB model weights and coordinate attention calculations across both GPUs—requiring absolutely no complex `torchrun` syntax!
```bash
python3 generate_vllm.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "Two anthropomorphic cats in comfy boxing gear fighting intensely on stage" \
    --tensor_parallel_size 2
```

#### Extreme Distributed Scaling (4x or 8x GPUs)
If deploying on large clusters (like an 8x H100 node), just match the parallel size to your GPU count to slash the 25-minute generation time down to a fraction of the time:
```bash
python3 generate_vllm.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "An epic fluid drone shot flying over a frozen mountainside at dawn" \
    --tensor_parallel_size 8
```

---

### Script Parameters 
You can customize the generation natively via these flags:
* `--model`: The HuggingFace Repo ID (e.g., `Wan-AI/Wan2.2-T2V-A14B-Diffusers`) *(Required)*.
* `--prompt`: The text guide for your video *(Required)*.
* `--resolution`: The output resolution (default: `1280x720`).
* `--frames`: Total video output frames (default: `81`).
* `--tensor_parallel_size`: How many GPUs to distribute inference across (default: `1`).
