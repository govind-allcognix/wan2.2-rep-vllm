import argparse
from vllm_omni.entrypoints.omni import Omni
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
    
    # Initialize the specialized vLLM Omni Engine specifically built for Diffusion
    engine = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,           
        trust_remote_code=True,
        gpu_memory_utilization=0.95   
    )
    
    print(f"\n[+] Generating video for prompt: '{args.prompt}'")
    print(f"[+] Settings: {args.resolution} @ {args.frames} frames")
    
    outputs = engine.generate(args.prompt)
    
    print("\n[+] Generation complete!")
    print("Result:", outputs)

if __name__ == "__main__":
    main()
