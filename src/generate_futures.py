import os
import gc
import json
import time
import numpy as np
import torch

from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import load_image, export_to_video
from huggingface_hub import snapshot_download
from tesseract.utils import crop_and_resize_frames

PROMPT = "Move the cup near bottle Franka Emika Panda"
IMAGE_PATH = "asset/images/majo.jpg"
BASE_MODEL = "THUDM/CogVideoX-5b-I2V"
LORA_REPO = "anyeZHY/tesseract"
LORA_SUBFOLDER = "tesseract_v01e_rgb_lora"
OUTPUT_DIR = "mirage_runs"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88]
HEIGHT, WIDTH, FPS, STEPS, GUIDANCE = 480, 640, 8, 50, 7.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.set_grad_enabled(False)
print("MIRAGE-X: controlled test-time imagination")
load_start = time.time()
pipe = CogVideoXImageToVideoPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float16).to("cuda")
if hasattr(pipe.transformer.patch_embed, "pos_embedding"):
    del pipe.transformer.patch_embed.pos_embedding
pipe.transformer.patch_embed.use_learned_positional_embeddings = False
pipe.transformer.config.use_learned_positional_embeddings = False
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
snapshot = snapshot_download(repo_id=LORA_REPO, local_dir_use_symlinks=False)
lora_path = os.path.join(snapshot, LORA_SUBFOLDER)
pipe.load_lora_weights(lora_path, adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0])
load_seconds = time.time() - load_start
image = load_image(IMAGE_PATH)
image = crop_and_resize_frames([np.array(image)], (HEIGHT, WIDTH))[0]
image = torch.from_numpy(np.array(Image.fromarray(image))).to("cuda") / 255.0
image = image.permute(2, 0, 1).unsqueeze(0)
manifest = {"experiment":"MIRAGE-X Experiment 1","base_model":BASE_MODEL,"world_model_adapter":f"{LORA_REPO}/{LORA_SUBFOLDER}","prompt":PROMPT,"image":IMAGE_PATH,"resolution":[WIDTH,HEIGHT],"steps":STEPS,"guidance_scale":GUIDANCE,"load_seconds":load_seconds,"runs":[]}
for i, seed in enumerate(SEEDS):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()
    result = pipe(image=image,prompt=PROMPT,guidance_scale=GUIDANCE,use_dynamic_cfg=True,height=HEIGHT,width=WIDTH,num_inference_steps=STEPS,generator=generator)
    runtime = time.time()-start
    frames=result.frames[0]
    filename=f"future_{i:02d}_seed_{seed}.mp4"
    export_to_video(frames,os.path.join(OUTPUT_DIR,filename),fps=FPS)
    manifest["runs"].append({"future_index":i,"seed":seed,"file":filename,"runtime_seconds":runtime,"peak_allocated_gb":torch.cuda.max_memory_allocated()/(1024**3),"peak_reserved_gb":torch.cuda.max_memory_reserved()/(1024**3),"num_frames":len(frames)})
    with open(os.path.join(OUTPUT_DIR,"manifest.json"),"w") as f: json.dump(manifest,f,indent=2)
