# -*- coding: utf-8 -*-
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        self.l1 = nn.L1Loss()
        self.l2 = nn.L1Loss()
        self.l3 = nn.L1Loss()
    
    def forward(self, pred1, pred2, pred3, y1, y2, y3):
        l1 = self.l1(pred1, y1)
        l2 = self.l2(pred2, y2)
        l3 = self.l3(pred3, y3)
        loss = (l1 + l2 + l3)/3.0
        return loss