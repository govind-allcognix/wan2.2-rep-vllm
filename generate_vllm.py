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
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    engine = Omni(
        model=args.model,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=args.tensor_parallel_size),
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
        
        video_payload = {
            "prompt": args.prompt,
            "multi_modal_data": {"video": {"num_frames": args.frames, "height": height, "width": width}}
        }
        outputs = engine.generate(video_payload, sampling_params_list=sampling_params)
    except Exception as e:
        print("[!] Exception formatting multimodal payload:", e)
        outputs = engine.generate(args.prompt)
    
    print("\n[+] Generation complete!")
    
    from diffusers.utils import export_to_video
    import numpy as np
    import torch

    result_obj = outputs[0]
    if getattr(result_obj, 'request_output', None):
        inner = result_obj.request_output
        if isinstance(inner, list) and len(inner) > 0:
            result_obj = inner[0]
        else:
            result_obj = inner
            
    frames = None
    if hasattr(result_obj, 'images') and result_obj.images:
        images_list = result_obj.images
        if len(images_list) == 1 and isinstance(images_list[0], dict):
            frames = images_list[0].get("frames") or images_list[0].get("video")
        elif len(images_list) == 1 and isinstance(images_list[0], tuple):
            frames = images_list[0][0]
        else:
            frames = images_list

    if not frames:
        print("[!] No video frames found in output.")
        return

    print(f"[+] Total frames extracted: {len(frames) if isinstance(frames, (list, tuple)) else 'Tensor'}")

    def _normalize_frame(frame):
        if isinstance(frame, torch.Tensor):
            frame_tensor = frame.detach().cpu()
            if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
                frame_tensor = frame_tensor[0]
            if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
                frame_tensor = frame_tensor.permute(1, 2, 0)
            if frame_tensor.is_floating_point():
                frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
            return frame_tensor.float().numpy()
        if isinstance(frame, np.ndarray):
            frame_array = frame
            if frame_array.ndim == 4 and frame_array.shape[0] == 1:
                frame_array = frame_array[0]
            if np.issubdtype(frame_array.dtype, np.integer):
                frame_array = frame_array.astype(np.float32) / 255.0
            return frame_array
        try:
            from PIL import Image
            if isinstance(frame, Image.Image):
                return np.asarray(frame).astype(np.float32) / 255.0
        except ImportError:
            pass
        return frame

    if isinstance(frames, (torch.Tensor, np.ndarray)):
        video_array = [_normalize_frame(f) for f in frames]
    elif isinstance(frames, list):
        video_array = [_normalize_frame(frame) for frame in frames]
    else:
        video_array = frames
        
    # Unpack the 4D Tensor into a standard sequence of 3D image arrays
    if isinstance(video_array, list) and len(video_array) == 1 and isinstance(video_array[0], np.ndarray):
        if video_array[0].ndim == 4:
            video_array = list(video_array[0])
        elif video_array[0].ndim == 5:
            video_array = list(video_array[0][0])
            
    export_to_video(video_array, "output.mp4", fps=16)
    print(f"[+] SAVED physically to: current directory as output.mp4")

if __name__ == "__main__":
    main()
