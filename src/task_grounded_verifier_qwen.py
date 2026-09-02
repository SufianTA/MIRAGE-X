import argparse, json, re, math
from pathlib import Path
import cv2, torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

RUBRIC = '''You are a strict robotics world-model evaluator.
Task: Move the cup near the bottle using the Franka Emika Panda robot.
You are judging ordered frames sampled from ONE predicted future video. The PRIMARY question is whether the robot causally and visibly moves the intended cup closer to the bottle.
Do NOT reward a rollout merely because objects remain visually consistent. Penalize no clear robot-caused cup movement, wrong-object interaction, incomplete final state, teleportation, collision/interpenetration, unrealistic or failed grasp, and severe robot/object deformation.
Score ONLY from visible evidence. If evidence is ambiguous, score conservatively.
Return ONLY valid JSON with: task_completion 0-4, physical_plausibility 0-2, manipulation_feasibility 0-2, object_continuity 0-1, temporal_consistency 0-1, major_failure boolean, failure_tags list, confidence 0-1, brief_rationale string.
task_completion: 0=no visible evidence; 1=ambiguous/minimal progress; 2=clear progress but incomplete; 3=likely successful; 4=clearly successful.
major_failure MUST be true for any decisive failure such as no meaningful task progress, wrong object, impossible grasp, severe collision/interpenetration, or clearly incomplete manipulation. Confidence MUST NOT increase quality score.'''

def extract_frames(path,n=7):
    cap=cv2.VideoCapture(str(path)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); idxs=[round(i*(total-1)/(n-1)) for i in range(n)] if total>1 else [0]*n; frames=[]
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
        if ok: frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    cap.release(); return frames

def parse_json(text):
    m=re.search(r'\{.*\}',text,flags=re.S)
    if not m: raise ValueError('No JSON object found')
    return json.loads(m.group(0))

def clamp(x,a,b): return max(a,min(b,x))
def quality_score(j):
    base=clamp(float(j['task_completion']),0,4)+clamp(float(j['physical_plausibility']),0,2)+clamp(float(j['manipulation_feasibility']),0,2)+clamp(float(j['object_continuity']),0,1)+clamp(float(j['temporal_consistency']),0,1)
    if bool(j.get('major_failure',False)): base-=2
    return clamp(base,0,10)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--videos',default='mirage_runs'); ap.add_argument('--out',default='mirage_strict_critic.json'); ap.add_argument('--passes',type=int,default=3); ap.add_argument('--model',default='Qwen/Qwen2.5-VL-7B-Instruct'); args=ap.parse_args()
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,torch_dtype='auto',device_map='auto'); processor=AutoProcessor.from_pretrained(args.model)
    videos=sorted(Path(args.videos).glob('future_*.mp4')); all_results=[]
    for video in videos:
        frames=extract_frames(video,7); passes=[]
        for p in range(args.passes):
            content=[{'type':'text','text':RUBRIC}]
            for k,img in enumerate(frames): content += [{'type':'text','text':f'Ordered frame {k+1} of {len(frames)}:'},{'type':'image','image':img}]
            content.append({'type':'text','text':f'This is independent judge pass {p+1}. Evaluate conservatively and return JSON only.'})
            messages=[{'role':'user','content':content}]; text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True); image_inputs,video_inputs=process_vision_info(messages)
            inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt').to(model.device)
            with torch.no_grad(): ids=model.generate(**inputs,max_new_tokens=320,do_sample=False)
            trimmed=[o[len(i):] for i,o in zip(inputs.input_ids,ids)]; j=parse_json(processor.batch_decode(trimmed,skip_special_tokens=True)[0]); j['quality_score']=quality_score(j); j['pass']=p+1; passes.append(j)
        scores=[x['quality_score'] for x in passes]; tasks=[float(x['task_completion']) for x in passes]; confs=[float(x.get('confidence',0)) for x in passes]
        all_results.append({'file':video.name,'passes':passes,'mean_score':sum(scores)/len(scores),'min_score':min(scores),'max_score':max(scores),'score_range':max(scores)-min(scores),'mean_task_completion':sum(tasks)/len(tasks),'mean_confidence':sum(confs)/len(confs),'major_failure_majority':sum(bool(x.get('major_failure')) for x in passes)>=math.ceil(args.passes/2)})
    ranked=sorted(all_results,key=lambda x:(x['mean_score'],x['mean_task_completion'],x['mean_confidence']),reverse=True)
    scaling=[]
    for k in [1,2,4,8]:
        subset=all_results[:min(k,len(all_results))]; best=max(subset,key=lambda x:(x['mean_score'],x['mean_task_completion'],x['mean_confidence'])); scaling.append({'K':k,'selected':best['file'],'score':best['mean_score']})
    output={'critic_model':args.model,'judge_passes':args.passes,'rubric':'failure-aware task-dominant v2','results':all_results,'ranking':[{'rank':i+1,'file':r['file'],'mean_score':r['mean_score'],'task_completion':r['mean_task_completion'],'score_range':r['score_range'],'major_failure_majority':r['major_failure_majority']} for i,r in enumerate(ranked)],'selection_scaling':scaling,'adaptive':{'K':8,'selected':ranked[0]['file'],'score':ranked[0]['mean_score'],'decision':'ABSTAIN if no candidate clears task-grounded threshold'}}
    Path(args.out).write_text(json.dumps(output,indent=2))
if __name__=='__main__': main()
