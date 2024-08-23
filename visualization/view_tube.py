import numpy as np
import json
import sys
import os
import pickle
from pathlib import Path
from argparse import ArgumentParser
import cv2
from tqdm import tqdm
sys.path.append(os.path.abspath("."))
from box_tools import norm_box_into_absolute, bbox_normalized

_color_map = None
_CTABLE_FILE = Path(os.path.dirname(os.path.realpath(__file__)))/"color_map.json"

used_labels =[
    "Ped", "Car", "Cyc", "Mobike", "SmalVeh",
    "MedVeh", "LarVeh", "Bus", "EmVeh", "TL"
]
import cv2

def plot_box(frame, box, text, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Add text next to the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (x1, y1 - 10)  # Adjust the text position based on your preference
    font_scale = 0.6
    color = color
    thickness = 2
    cv2.putText(
        frame, text, org, font, 
        font_scale, color, thickness, cv2.LINE_AA
    )

def plot_track(source_video, output_video_path, tracklets):
    global _color_map
    if _color_map is None:
        with open(_CTABLE_FILE,"r") as f:
            _color_map = json.load(f)
        
    cap = cv2.VideoCapture(str(source_video))
    src_fps = int(cap.get(cv2.CAP_PROP_FPS))
    ori_frames = []
    fid = 1
    while True:
        ret, frame = cap.read()
        cv2.putText(
            frame, f"{fid}", (100, 100), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
            color=(0,0,0), thickness=2
        )
        fid += 1
        if not ret:
            break
        ori_frames.append(frame)
    cap.release()

    for tid, tubes in enumerate(tqdm(tracklets)):
        for i in range(len(tubes['frames'])):
            frame_num = tubes['frames'][i] - 1
            cls = used_labels[tubes['label_id']]
            box = norm_box_into_absolute(
                bbox=bbox_normalized(tubes['boxes'][i], 840, 600),
                img_w=1920,
                img_h=1280
            )
            
            plot_box(
                ori_frames[frame_num], box, 
                text = f"{cls}{tid}_{tubes['scores'][i]:.4f}",
                color = _color_map[tid%50]    
            )

            
    height, width, _ = ori_frames[0].shape
    out = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), 
        src_fps, (width, height)
    )
    print(f"Write the frames into the {output_video_path}")
    for frame in tqdm(ori_frames):
        out.write(frame)
    
    out.release()

if __name__ == "__main__":
    
    
    parser = ArgumentParser()
    parser.add_argument("--video_root", type=Path, default=Path("/datasets")/"roadpp"/"test_videos")
    parser.add_argument("--seq", type=int)
    parser.add_argument("--output_root", type=Path, default=Path("demo"))
    parser.add_argument("--tracklet", type=Path, default=Path("submit")/"cb"/"cb_tubes.pkl")
    args = parser.parse_args()
    seq = "val_" + f"{args.seq}".zfill(5)
    source_video = args.video_root/f"{seq}.mp4" 
    t = None
    with open(args.tracklet, "rb") as fff:
        t = pickle.load(fff)['agent']
    tracklets = t[seq]
    print(source_video)
    plot_track(
        source_video=source_video, 
        output_video_path=args.output_root/f"{seq}.mp4",
        tracklets=tracklets
    )