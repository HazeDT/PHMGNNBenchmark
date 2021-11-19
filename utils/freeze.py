
from collections import Iterable

def set_freeze_by_id(model, layer_num_last):
    for param in model.parameters():
        param.requires_grad = False
    child_list = list(model.children())[-layer_num_last:]
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        for param in child.parameters():
            param.requires_grad = True