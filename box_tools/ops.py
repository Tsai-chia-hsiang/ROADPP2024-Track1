import torchvision.ops as ops
import torch

def pairwise_min_area(A:torch.Tensor)->torch.Tensor:
    row_area = A.repeat(A.size(0), 1)
    column_area = A.unsqueeze(1).repeat(1, A.size(0))
    return torch.min(row_area, column_area)

def remove_dup(boxes:torch.Tensor, iou_thr:float, fid:int=None) -> torch.Tensor:
  
    ordered = torch.argsort(boxes[:, 4], descending=True)
    boxes = boxes[ordered]
    inter, _ = ops.boxes._box_inter_union(
        boxes1=boxes[:, :4], 
        boxes2=boxes[:, :4]
    )
    inter /= pairwise_min_area(ops.box_area(boxes=boxes[:, :4]))
  
    out = []
    keep = [_ for _ in range(boxes.size(0))]
    for i in range(inter.size(0)):
        if i not in out:
            dup = torch.where(inter[i, i+1:] > iou_thr)[0] + i + 1
            if len(dup):
                out  += dup.cpu().tolist()
    out = set(out)
    keep = list(filter(lambda x:x not in out, keep))
    ret =  boxes[torch.tensor(keep, device=boxes.device, dtype=torch.long)]

    return ret
    