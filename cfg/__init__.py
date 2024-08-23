from .default import _C
import os
from yacs.config import CfgNode as CN

def load_config(yaml_file: os.PathLike) -> CN:
    
    with open(yaml_file, "r") as s:
        setting = CN.load_cfg(s)
    if 'dataset' in setting and 'target_seq' not in setting.dataset:
        setting.dataset.target_seq = []

    _C.merge_from_other_cfg(setting) 
    return _C