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
    # We must restrict vLLM's intrinsic cache to ~65-70% of the GPU so 
    # the 3D VAE decoder has enough raw VRAM left over at the end to 
    # decompress the 1280x720x81 latent tensors into raw pixels!
    engine = Omni(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,           
        trust_remote_code=True,
        gpu_memory_utilization=0.65,
        vae_use_slicing=True,
        vae_use_tiling=True
    )
    
    print(f"\n[+] Generating video for prompt: '{args.prompt}'")
    print(f"[+] Settings Requested: {args.resolution} @ {args.frames} frames")
    
    # Parse requested resolution
    width, height = map(int, args.resolution.lower().split('x'))
    
    try:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        # Inject exact video dimensions directly into the specialized parameter class
        sampling_params = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=args.frames,
            num_inference_steps=20 # Default step count for flow-matching
        )
        outputs = engine.generate(args.prompt, sampling_params_list=sampling_params)
    except Exception as e:
        print("[!] Could not inject exact height/width custom parameters:", e)
        print("[!] Generating via Model Constants instead...")
        outputs = engine.generate(args.prompt)
    
    print("\n[+] Generation complete!")
    
    # Extract the payload from vLLM's custom request object
    result_obj = outputs[0].request_output[0] if getattr(outputs[0], 'request_output', None) else outputs[0]
    
    if hasattr(result_obj, 'images') and result_obj.images:
        frames = result_obj.images
        print(f"[+] Total frames generated: {len(frames)}")
        
        import torchvision.io
        import numpy as np
        
        # Convert PIL Images to Tensor (T, H, W, C)
        video_tensor = torch.stack([torch.from_numpy(np.array(img)) for img in frames])
        torchvision.io.write_video(
            "output.mp4", 
            video_tensor, 
            fps=16, 
            video_codec="libx264"
        )
        print("[+] SAVED physically to: current directory as output.mp4")
    else:
        print("[!] No images found in the output object:", outputs)

if __name__ == "__main__":
    main()
