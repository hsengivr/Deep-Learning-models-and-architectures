import torch

def generalized_intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
  """
  Calculates generalized intersection over union of two bboxes
  Parameters:
    boxes_pred : bbox predictions usually the output of a neural network, either during training or at inference time of size (batch, 4) 
    boxes_labels : ground truth bboxes of size (batch, 4)
    box_format : midpoint (x,y,w,h) or corners (x1, y1, x2, y2)
  Returns:
    Generalized Intersection over union for all examples
  References:
    https://giou.stanford.edu/
    https://giou.stanford.edu/GIoU.pdf
  """
  if box_format=='midpoint':
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

  elif box_format == "corners":
    box1_x1 = boxes_preds[..., 0:1]
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_labels[..., 0:1]
    box2_y1 = boxes_labels[..., 1:2]
    box2_x2 = boxes_labels[..., 2:3]
    box2_y2 = boxes_labels[..., 3:4]

    assert (box1_x2, box1_y2) > (box1_x1, box1_y1)
    box1_x1 = torch.min(box1_x1, box1_x2)
    box1_y1 = torch.min(box1_y1, box1_y2)
    box1_x2 = torch.max(box1_x1, box1_x2)
    box1_y2 = torch.max(box1_y1, box1_y2)
    
    # calculate area of ground truth and predicted bbox
    gt_boxarea =  abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1)) 
    pred_boxarea = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))

    # IOU area of intersection
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y1, box2_y1)

    I = (x2 - x1) * (y2 - y1) if ((x2>x1) and (y2>y1)) else 0

    # Finding the coordinate of smallest enclosing box Bc
    x1_con = torch.min(box1_x1, box2_x1)
    y1_con = torch.min(box1_y1, box2_y1)
    x2_con = torch.max(box1_x2, box2_x2)
    y2_con = torch.max(box1_y1, box2_y1)

    # find enclosing box area
    C = abs((x2_con - x1_con)*(y2_con - y1_con))
    
    # area of union 
    U = (gt_boxarea + pred_boxarea - I)

    IOU = I / U
    GIOU = IOU - ((C - U)/ C)

    # loss_iou = 1- IOU
    # loss_giou = 1-GIOU
    return GIOU
