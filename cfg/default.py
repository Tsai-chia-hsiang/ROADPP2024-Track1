from yacs.config import CfgNode as CN

_C = CN()
_C.detector = CN()
_C.detector.device = 0
_C.detector.weights = ""
_C.detector.threshold = 0.007

_C.trackers = []

_C.dataset = CN()
_C.dataset.video_root = ""
_C.dataset.video_postfix = "mp4"
_C.dataset.target_seq = []