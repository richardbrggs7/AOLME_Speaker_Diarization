#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:37:45 2023
linearClassifier.py
@author: richard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class linearClassifier(nn.Module):
    # def __init__(self, numLearners, learnerIn, learnerMid, learnerOut, device, extra=0):
    def __init__(self, inSize, midSize):
        super(linearClassifier, self).__init__()
        self.inSize = inSize
        self.midSize = midSize

        self.linear0 = nn.Linear(self.inSize, self.midSize)
        self.linear1 = nn.Linear(self.midSize, int(self.midSize/2))
        # self.linear2 = nn.Linear(int(self.midSize/2), int(self.midSize/4))
        self.classify = nn.Linear(int(self.midSize/2), 5)

    def forward(self, x):
        x = F.relu(self.linear0(x))
        x = F.relu(F.dropout(self.linear1(x)))
        # x = F.relu(F.dropout(self.linear2(x)))
        x = F.relu(self.classify(x))

        x = F.softmax(x, 1)
        return x
