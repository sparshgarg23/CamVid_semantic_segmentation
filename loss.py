# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:21:39 2022

@author: Admin
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

def flatten(tensor):
    """
    convert (N,C,D,H,W)->(C,N*D*H*W)
    """
    C=tensor.size(1)
    axis_order=(1,0)+tuple(range(2,tensor.dim()))
    transposed=tensor.permute(axis_order)
    return transposed.contiguous().view(C,-1)

class DiceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.eps=1e-5
    def forward(self,output,target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        
        output=F.softmax(output,dim=1)
        output=flatten(output)
        target=flatten(target)
        
        intersect=(output*target).sum(-1)
        denom=(output+target).sum(-1)
        dice=intersect/denom
        dice=torch.mean(dice)
        return 1-dice
    
        
        

    