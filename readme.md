## For [ROAD++@ECCV2024](https://sites.google.com/view/road-eccv2024/challenge?authuser=0)  Track1


## install :
### Environment
- OS: Ubuntu24.04

1. Create a python3.10.14 environment
2.  install __torch torchvision__ following [PyTorch](https://pytorch.org/get-started/locally/)
3. git clone this repo 
4.  pip install -r requirments.txt
5. 
    - Download the pretrained pedestrian ReID model ```MOT17-SBS-S50``` from Bot-SORT. 
        - The link can be find at [BoT-SORT](https://github.com/NirAharon/BoT-SORT/) readme file.
    - Download the pretrained vehicle ReID model ```BoT(R50-ibn)``` from __VehicleID Baseline__
        - The line can be find at fast reid [MODEL_ZOO](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)
    
    Put above 2 pretrained weights at ```./fast_reid/ckpt/```.

## First: Train the YOLOv10 detector
We using 1080Ti x 4 to train YOLOv10 from [ultralytics](https://www.ultralytics.com/)

The training settings can be found at the default arguments of [```parse_cmd_args()```](./train_detector.py#L70) of ```train_detector.py```

- training frame : validation frame $\approx$ 8 : 2
- **The training/validation frames are randomly split.

example :

```python train_detector.py --detector yolov10l.pt --device 0,1,2,3 --data_config ./detection/track1.yaml --project ./detection/ckpt --name baseline --epochs 5 --batch 3 --optimizer auto```

Then the ultralytics project will be saved at {--project}/{--name} (in above case : ```./detection/ckpt/baseline/```)
and the weights is at {--project}/{--name}/weight/best.pt

In our experiment, the best weights are from the epoch 2.

## Second: Tracking with BoTSORT
We use multiple semantic-based trackers to follow the detection results from YOLOv10. Each tracker is designed to associate with only the corresponding classes of detection results.

The setting file formation:
a yaml:
```
detector:
  device: 
    # due to the implementation of ultralytics, please use CUDA_VISIBLE_DEVICES to truely select device. currently only set it to 0 all the time

  weights: 
    # The weights path of YOLOv10 detector (e.g. ./detection/ckpt/baseline/weights/best.pt)
  threshold: 
    # minimum detection threshold. Please notice that the value should give all the trackers a chance to associate low confident detections.
    (i.e. Don’t set it too high to ensure that the low threshold is still used by some trackers)

trackers:
  # a list of tracker, each tracker is in BotSORT args format

dataset:
  video_root: # the root of the videos that wants to track.

```

All the threshold settings for BoTSORT used by each tracker can be found in the [```./configs/baseline.yaml```](./configs/baseline.yaml)

- __ped__ (BoTSORT): 
    
    The classes that may containing pedestrian semantic information:
    
    - 0: Ped
    - 2: Cyc
    - 3: Mobike

    For this tracking branch, we use the pretrained pedestrian ReID model from BoTSORT as the reid model.
    (mot17_sbs_S50.pth)

- __veh__ (BoTSORT): 
    
    The classes that are vehicle-like.
    
    - 1 : Car
    - 4 : SmalVeh
    - 5 : MedVeh
    - 6 : LarVeh
    - 8 : EmVeh
    
    For this tracking branch, we use the pretrained vehicle ReID model from BoTSORT as the reid model.
    (vehicleid_bot_R50-ibn.pth)

- __bus__ (ByteTrack): 

    The bus class was separated from the vehicle branch because the pretrained vehicle ReID model doesn't perform very well on buses. Additionally, buses require different confidence thresholds for BOTSort compared to other vehicles.
    
    - 7: Bus

    For this tracking branch, we just use ByTrack and do not use any ReID model.

- __TL__ (ByteTrack): 
    
    Traffic Light.
    
    - 9 : TL
    
    For this tracking branch, we just use ByTrack and do not use any ReID model.


### Tracking wtih baseline settings: 

```python tracking.py --config ./config/baseline.yaml --pkl_dir path_to_save_result```

- To track some specific val sequences please set it at ```./config/baseline.yaml``` [```dataset: target_seq```](./configs/baseline.yaml#80) parts
    - If tracking the specific val sequences, you can pass the arg ```--demo ./demo``` to do visualization of tracking results. It will wirte the video with tracked bboxes for each frame.

### Tracking with specific semantic (ped, veh, bus, TL):
```python tracking.py --config ./config/each_semantic/{SEMANTIC}.yaml --pkl_dir path_to_save_result```

Also, the sepcific val sequence and the visualization can be set as the above method. 

e.g. 
1. set the target sequence to 100 at [targe_seq](configs/each_semantic/veh.yaml#4)  

2. ```python tracking.py --config ./config/each_semantic/veh.yaml --pkl_dir ./submit/veh_only --demo ./demo``` 

## Acknowledgments

This project utilizes code and resources from the following repositories:

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT/) : The code base of ```./tracker``` 
- [fast Reid](https://github.com/JDAI-CV/fast-reid) To re-identify objects during tracking and prevent ID switches, the same as how [BoT-SORT](https://github.com/NirAharon/BoT-SORT/) operates.

We deeply appreciate the work of these developers and their contributions to the open-source community.
