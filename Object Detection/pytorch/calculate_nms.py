# multiple bboxes for one object . cleaned up using nms 
# take out the bbox with highest score and compare it with other boxes by calculating iou
# if iou of other box > certain threshold for bbox with highest score then remove the other box with lower probability 
# - if we have multiple class then do nms seperate for each class

import torch 
from calculate_iou import intersection_over_union

def non_max_suppression(predictions, iou_threshold,prob_threshold, box_format='corners'):
  """
  Does Non Max Suppression on given bboxes
    Parameters:
        predictions : list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold : threshold where predicted bboxes is correct
        prob_threshold : threshold to remove predicted bboxes (independent of IoU) 
        box_format : "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
  """
  # predictions = [[i, 0.9, x1,y1,x2,y2]]

  assert type(predictions) == list
  
  # consider bbox with probability values>certain threshold
  bboxes = [box for box in predictions if box[1] > prob_threshold ]
  # sort bboxes based on probabilities
  bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
  bboxes_after_nms=[]

  while bboxes:
    chosen_box = bboxes.pop(0)
    # 
    bboxes = [box for box in bboxes  
                  if box[0]!=chosen_box[0]
                  or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold]
    bboxes_after_nms.append(chosen_box)
  return bboxes_after_nms
