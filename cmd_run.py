from vllm_omni.entrypoints.omni import Omni
import torch

def main():
    width = 1280
    height = 720
    model = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    prompt = "A cinematic video of a cyberpunk city at night with neon lights reflecting in puddles"
    tensor_parallel_size = 1
    frames = 81

    print(f"Loading vLLM-Omni Diffusion Engine across {tensor_parallel_size} GPU(s)...")
    
    engine = Omni(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        sp_size=tensor_parallel_size,
        enforce_eager=True,           
        trust_remote_code=True,
        gpu_memory_utilization=0.70,
        vae_use_slicing=True,
        vae_use_tiling=True
    )
    
    print(f"\n[+] Generating video for prompt: '{prompt}'")
    print(f"[+] Settings Requested: {width}x{height} @ {frames} frames")
    
    try:
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        sampling_params = OmniDiffusionSamplingParams(
            height=height,
            width=width,
            num_frames=frames,
            num_inference_steps=20
        )
        outputs = engine.generate(prompt, sampling_params_list=sampling_params)
    except Exception as e:
        print("[!] Could not inject exact height/width custom parameters:", e)
        print("[!] Generating via Model Constants instead...")
        outputs = engine.generate(prompt)
    
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
            "/workspace/output.mp4", 
            video_tensor, 
            fps=16, 
            video_codec="libx264"
        )
        print("[+] SAVED physically to: /workspace/output.mp4")
    else:
        print("[!] No images found in the output object:", outputs)

if __name__ == "__main__":
    main()
