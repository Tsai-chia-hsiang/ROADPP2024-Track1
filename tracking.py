import os
import sys
sys.path.append(os.path.abspath("./tracker"))
sys.path.append(os.path.abspath("./fast_reid"))
from pathlib import Path
import shutil
from typing import Callable
import gc
import zipfile
from argparse import ArgumentParser
from argparse import Namespace
import numpy as np
import pickle
from yacs.config import CfgNode as CN
import torch
from ultralytics.engine.results import Results, Boxes
from ultralytics import YOLO
from tqdm import tqdm
import cv2

from cfg import load_config
from tracker.botsort import BoTSORT
from box_tools import tube_change_axis, out_of_range
from box_tools.ops import remove_dup
from visualization import plot_box, plot_track, used_labels

np.random.seed(891122)
C_MAP = np.random.randint(0, 256, (100, 3)).tolist()
CLS_C_MAP = np.random.randint(0, 256, (10, 3)).tolist()

def tube_interpolation(tube):
    frames = tube['frames']
    scores = tube['scores']
    boxes = tube['boxes']
    
    interpolated_frames = np.arange(frames[0], frames[-1] + 1)  
    interpolated_scores = np.interp(interpolated_frames, frames, scores)  
    interpolated_boxes = np.empty((len(interpolated_frames), 4))  
    
    for i, axis in enumerate([0, 1, 2, 3]):
        interpolated_boxes[:, i] = np.interp(interpolated_frames, frames, boxes[:, axis])
    
    tube['boxes'] = interpolated_boxes
    tube['scores'] = interpolated_scores
    tube['frames'] = interpolated_frames

@torch.no_grad()
def track_a_seq(detector:YOLO, seq:Path, trackers:list[dict[str, dict]], low_thr:float, pbar:tqdm, debug=False) -> list[dict]:
    
    def build_sorters(fps:int)->list[BoTSORT]:
        bot_trackers = []
        for v in trackers:
            bot_args = Namespace(**v['bot_args'])
            bot_args.device = detector.device
            bot_args.semantic = v['semantic']
            bot_args.cls = torch.tensor(v['cls_id'], dtype=torch.long, device=detector.device)
            bot_args.track_low_thresh =  low_thr if 'track_low_thresh' not in v['bot_args'] else v['bot_args']['track_low_thresh']
            bot_args.new_track_thresh = low_thr if 'new_track_thresh' not in v['bot_args'] else v['bot_args']['new_track_thresh']
            bot_trackers.append(BoTSORT(args=bot_args))#frame_rate=fps
        return bot_trackers
            
    if debug:
        log_frame = Path("debug")/f"{seq.stem}"
        if log_frame.exists():
            shutil.rmtree(log_frame)
        log_frame.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(seq))
    meta_data = {
        'fps':int(cap.get(cv2.CAP_PROP_FPS)),
        'nframes':int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    bot_sorter = build_sorters(fps=meta_data['fps'])
    group = np.arange(len(bot_sorter))
    tracklets = {}
    fid = 1
    img_plt, img_plt_det = None, None
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        r:Results = detector(frame, verbose=False, conf=low_thr)[0]
        
        detection_i:Boxes = r.boxes
        
        if debug:
            img_plt = frame.copy()
            img_plt_det = frame.copy()

        if detection_i.xyxy.size(0) == 0:
            # no detection
            continue

        pred_cls = detection_i.cls
        this_frame_det = 0
        this_frame_track = 0
        
        for g in group:
            
            this_group_cls = torch.nonzero(
                torch.isin(pred_cls, bot_sorter[g].corresponding_cls)
            ).squeeze()
            if this_group_cls.dim() == 0:
                this_group_cls = this_group_cls.unsqueeze(0)
            if this_group_cls.size(0) == 0:
                continue
            
            det = torch.hstack(
                [
                    detection_i.xyxy[this_group_cls], 
                    detection_i.conf[this_group_cls].unsqueeze(1),
                    detection_i.cls[this_group_cls].unsqueeze(1)
                ]
            )
            
            if 'dup_iou' in trackers[g]:
                det = remove_dup(det, iou_thr=trackers[g]['dup_iou'], fid=fid)
            
            this_frame_det += len(det)
            if debug:
                for di in det:
                    plot_box(
                        frame=img_plt_det, 
                        box=di[:4], text=f"{used_labels[int(di[5])]}_{di[4]:4f}",
                        color=CLS_C_MAP[int(di[5])]
                    )
            
            det = det.cpu().numpy()    
            online_targets = bot_sorter[g].update(det, frame, fid=fid)
            this_frame_track += len(online_targets)
            for t in online_targets:
                track_id = t.track_id
                x1, y1, x2, y2 = t.tlbr
                x1, y1 = out_of_range(x1, y1, r.orig_shape[1], r.orig_shape[0])
                x2, y2 = out_of_range(x2, y2, r.orig_shape[1], r.orig_shape[0])
                
                if track_id not in tracklets:
                   
                    tracklets[track_id] = {
                        'label_id':t.cls,
                        'boxes':np.array([[x1, y1, x2, y2]]),
                        'score':t.score,
                        'scores':np.array([t.score]),
                        'frames':np.array([fid])
                    }
                
                else:
                    tracklets[track_id]['scores'] = np.append(
                        tracklets[track_id]['scores'], t.score
                    )
                    tracklets[track_id]['boxes'] = np.append(
                        tracklets[track_id]['boxes'], [[ x1, y1, x2, y2]], 
                        axis=0
                    )
                    tracklets[track_id]['frames'] = np.append(
                        tracklets[track_id]['frames'], fid
                    )
                
                if debug:
                    plot_box(
                        frame=img_plt, box=[x1, y1, x2, y2], 
                        text=f"{t.cls}-{track_id}-{t.score:.4f}", 
                        color=C_MAP[track_id%100] 
                    )
    
        if debug and (this_frame_track > 0 or this_frame_det > 0):
            cv2.imwrite(
                f"./debug/{seq.stem}/{fid}.jpg",
                cv2.hconcat([img_plt_det, img_plt])
            )
        
        pbar.set_postfix(ordered_dict={'seq':f"{seq.stem}",'progress':f"{fid:3d}/{meta_data['nframes']:3d}"})
        fid += 1

    cap.release()
    
    agents = []
    for tuid, d in tracklets.items():
        tube_data = d.copy()
        tube_interpolation(tube_data)
        tube_change_axis(tube_data)
        tube_data['score'] = np.mean(tube_data['scores'])
        agents.append(tube_data.copy())
    
    return agents

def main(args:CN, cmd:Namespace):
    
    def zip_tube_file(file_path:Path):

        zip_path = file_path.parent/f"{file_path.stem}.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add the file to the zip archive
            zipf.write(file_path,file_path.name)

    def generate_seqs(naming_rule:Callable[[int|str],str])->list[Path]:
        seqs = None
        if len(args.dataset.target_seq) :
            seqs = [Path(args.dataset.video_root)/f"{naming_rule(i)}.{args.dataset.video_postfix}" for i in args.dataset.target_seq]
        else:
            seqs = Path(args.dataset.video_root).glob(f"*.{args.dataset.video_postfix}")
        return sorted(seqs, key=lambda x:int(x.stem.split("_")[-1]))   

    yolo = YOLO(args.detector.weights)
    yolo.to(torch.device(f"cuda:{args.detector.device}"))

    pbar = tqdm(generate_seqs(naming_rule=lambda x: f"{args.dataset.prefix}_{f'{x}'.zfill(5)}"))
    submit = {'agent':{}}
    for si in pbar:
        if not si.exists():
            raise FileNotFoundError(f"{si} doesn't exist")
        pbar.set_postfix(ordered_dict={"seq":f"{si.name}"})
        agent = track_a_seq(
            detector=yolo, seq=si, trackers=args.trackers, 
            pbar=pbar, low_thr=args.detector.threshold,
            debug=cmd.debug
        )
        submit['agent'][si.stem] = agent
        gc.collect()
        if cmd.demo != "" and len(args.dataset.target_seq):
            # print(len(submit['agent'][si.stem]))
            Path(cmd.demo).mkdir(exist_ok=True, parents=True)
            plot_track(
                source_video = si,
                output_video_path = \
                    Path(cmd.demo)/f"{cmd.pkl_dir.parts[-1]}_{si.name}",
                tracklets=submit['agent'][si.stem]
            )
    
    Path(cmd.pkl_dir).mkdir(parents=True, exist_ok=True)
    tube_file_name = cmd.pkl_dir/f"{cmd.pkl_dir.parts[-1]}_tubes.pkl"
    print(f"writing {tube_file_name} ..")
    with open(tube_file_name, 'wb+') as f:
        pickle.dump(submit, f)
    print("zipping ..")
    zip_tube_file(file_path=tube_file_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs")/"baseline.yaml")
    parser.add_argument("--pkl_dir", type=Path, default=Path("submit")/"baseline")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--demo", type=str, default="")
    cmd = parser.parse_args()
    args = load_config(cmd.config)
    main(args=args, cmd=cmd)
