from vllm_omni.entrypoints.omni import Omni
import torch

def main():
    width = 1280
    height = 720
    model = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    prompt = "A cinematic video of a cyberpunk city at night with neon lights reflecting in puddles"
    tensor_parallel_size = 2
    frames = 17

    print(f"Loading vLLM-Omni Diffusion Engine across {tensor_parallel_size} GPU(s)...")
    
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    engine = Omni(
        model=model,
        parallel_config=DiffusionParallelConfig(ulysses_degree=tensor_parallel_size),
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
            num_inference_steps=20,
            guidance_scale=5.0
        )
        
        # We MUST dynamically instruct the Multimodal vLLM Engine that we are building 
        # a Video, otherwise passing the prompt as a string triggers 1-frame Image generation!
        video_payload = {
            "prompt": prompt,
            "multi_modal_data": {"video": {"num_frames": frames, "height": height, "width": width}}
        }
        outputs = engine.generate(video_payload, sampling_params_list=sampling_params)
    except Exception as e:
        print("[!] Exception initializing generation payload:", e)
        outputs = engine.generate(prompt)
    
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
        
    # vLLM returns videos as a strictly packed 4D Numpy array inside a length-1 list.
    # We must unpack it into a flat list of F 3D frames so the MP4 encoder can read them!
    if isinstance(video_array, list) and len(video_array) == 1 and isinstance(video_array[0], np.ndarray):
        if video_array[0].ndim == 4:
            video_array = list(video_array[0])
        elif video_array[0].ndim == 5:
            video_array = list(video_array[0][0])
            
    export_to_video(video_array, "/workspace/output.mp4", fps=16)
    print(f"[+] SAVED physically to: /workspace/output.mp4")

if __name__ == "__main__":
    main()
