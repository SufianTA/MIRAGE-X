#!/usr/bin/env python3
import argparse, json, re, time
from pathlib import Path
import cv2
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

RUBRIC='''You are evaluating a robot world-model rollout for the instruction: "Move the cup near bottle Franka Emika Panda". You are given ordered frames from ONE predicted future. Score only what is visually supported. Return STRICT JSON with numeric scores in [0,2] for task_completion, object_continuity, physical_plausibility, manipulation_feasibility, temporal_consistency, confidence, plus failure_tags and a brief_rationale. Do not reward visual attractiveness. Penalize ambiguity rather than inventing success.'''

def sample_frames(video_path,n=7):
    cap=cv2.VideoCapture(str(video_path)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); idxs=[round(i*(total-1)/(n-1)) for i in range(n)]; frames=[]
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
        if not ok: raise RuntimeError(f'Failed reading frame {idx}')
        frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    cap.release(); return frames,idxs

def extract_json(text):
    m=re.search(r'\{.*\}',text,re.S)
    if not m: raise ValueError('No JSON object found')
    return json.loads(m.group(0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--videos',default='mirage_runs'); ap.add_argument('--out',default='mirage_critic.json'); ap.add_argument('--model',default='Qwen/Qwen2.5-VL-7B-Instruct'); ap.add_argument('--frames',type=int,default=7); args=ap.parse_args()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map='auto',attn_implementation='sdpa'); processor=AutoProcessor.from_pretrained(args.model); results=[]
    for video in sorted(Path(args.videos).glob('future_*.mp4')):
        frames,idxs=sample_frames(video,args.frames); content=[{'type':'image','image':im} for im in frames]+[{'type':'text','text':RUBRIC}]; messages=[{'role':'user','content':content}]; text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); image_inputs,video_inputs=process_vision_info(messages); inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt').to(model.device)
        t0=time.time()
        with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=350,do_sample=False)
        trimmed=[out[len(inp):] for inp,out in zip(inputs.input_ids,generated)]; score=extract_json(processor.batch_decode(trimmed,skip_special_tokens=True)[0]); score['total_10']=sum(float(score[k]) for k in ['task_completion','object_continuity','physical_plausibility','manipulation_feasibility','temporal_consistency']); score['video']=video.name; score['sampled_frames']=idxs; score['critic_seconds']=time.time()-t0; results.append(score)
    best=max(results,key=lambda r:(r['total_10'],r.get('confidence',0))); Path(args.out).write_text(json.dumps({'critic_model':args.model,'n_rollouts':len(results),'best_video':best['video'],'best_score':best['total_10'],'rollouts':results},indent=2))
if __name__=='__main__': main()
