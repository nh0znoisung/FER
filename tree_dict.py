from collections import OrderedDict
import torch 
from typing import List

def write(filename, txt):
    with open(filename, 'a', encoding = 'utf-8') as f:
        f.write(txt)
        f.close()

def customize_text(_key, _val, tab=0):
    _typ_val_string = get_type_string(_val)
    _typ_val = type(_val)
    
    txt = '\t'*tab + '|' + str(_key) + ": " + _typ_val_string + " "
    if _typ_val is int:
        txt += str(_val)
    elif _typ_val is float:
        txt += str(_val)
    elif _typ_val is str:
        txt += str(_val)
    elif _typ_val is list:
        txt += str(len(_val))
        # TODO: write whole list if len < 50
        if len(_val) < 50:
            txt += "\n" + '\t'*tab + '|' + str(_val)
    elif _typ_val is dict:
        txt += str(len(_val.items()))
    elif _typ_val is torch.Tensor:
        txt += str(_val.size())

    txt += '\n'
    return txt
def clear_file(filename):
    with open(filename, 'w') as f:
        f.close()

def get_type_string(var):
    return var.__class__.__name__

    
# Using depth first search
def tree_dict(_key, _val, filename, tab=0):
    write(filename, customize_text(_key, _val, tab))
    if type(_val) is dict or type(_val) is OrderedDict:
        for key, val in _val.items():
            tree_dict(key, val, filename, tab+1)

def tree_dict_util(val, filename):
    clear_file(filename)
    tree_dict("root", val, filename) 



FILENAME = "a.txt"
model = torch.load("./pretrain_models/dan_affectnet7.pth", map_location="cpu")

tree_dict_util(model, FILENAME)