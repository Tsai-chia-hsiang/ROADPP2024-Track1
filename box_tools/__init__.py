import numpy as np

def norm_box_into_absolute(bbox, img_w, img_h):
    return bbox * np.array([img_w, img_h, img_w, img_h])

def bbox_normalized(bbox, img_w, img_h):
    return bbox / np.array([img_w, img_h, img_w, img_h])
    

def tube_change_axis(tube, orig_shape=(1280, 1920), submit_shape=(600, 840)):
        
    ori_h, ori_w = orig_shape
    new_h, new_w = submit_shape
    
    tube['boxes'] = np.array(
        [
            norm_box_into_absolute(
                bbox_normalized(box, ori_w, ori_h), 
                new_w, new_h
            ) 
            for box in tube['boxes']
        ]
    )

def out_of_range(x:float, y:float, max_x:float|int, max_y:float|int)->tuple[float, float]:
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y
