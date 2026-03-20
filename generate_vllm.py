import argparse
from vllm import LLM, SamplingParams
import torch

def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Video Generation using vLLM-Omni")
    parser.add_argument("--model", type=str, required=True, help="Path to the model weights directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to split the model across")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Video resolution")
    args = parser.parse_args()

    print(f"Loading vLLM-Omni Diffusion Engine across {args.tensor_parallel_size} GPU(s)...")
    
    # Initialize the vLLM Engine
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,           # Often required for beta multimodal architectures
        trust_remote_code=True,
        gpu_memory_utilization=0.95   # Maximize VRAM usage
    )
    
    # Note: Omni's extended SamplingParams for video might take specific kwargs for W/H/Frames
    # Refer to vllm-omni docs for exact kwarg mapping. This is a generic blueprint.
    sampling_params = SamplingParams(
        temperature=0.0,
        # Currently passing extras as mock implementation; adjust per omni API specs
    )
    
    print(f"\n[+] Generating video for prompt: '{args.prompt}'")
    print(f"[+] Settings: {args.resolution} @ {args.frames} frames")
    
    outputs = llm.generate([args.prompt], sampling_params)
    
    print("\n[+] Generation complete!")
    for output in outputs:
        # In vllm-omni, outputs hold the multimodal output tensor or saved path
        print("Result:", output)

if __name__ == "__main__":
    main()
