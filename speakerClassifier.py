#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:32:43 2022
speakerClassifier.py
@author: richard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class speakerClassifier(nn.Module):
    def __init__(self, numLearners, learnerIn, learnerMid, learnerOut, extra=0):
        super(speakerClassifier, self).__init__()
        self.speakerIn = learnerIn
        self.speakerMid = learnerMid
        self.speakerOut = learnerOut

        self.activator = nn.ReLU()
        self.numLearners = numLearners
        self.learners = nn.ParameterList()
        self.extra = extra

        for i in range(numLearners):
            self.learners.append(nn.Sequential(
                    nn.Linear(self.speakerIn, self.speakerMid),
                    self.activator,
                    nn.Linear(self.speakerMid, self.speakerOut),
                    nn.Dropout(p=0.2),
                    # nn.BatchNorm1d(self.speakerOut),
                    self.activator)
                )

        if extra != 0:
            self.linear0 = nn.Linear(self.speakerOut*self.numLearners+extra,
                                     self.speakerOut)
            # self.norm0 = nn.BatchNorm1d(self.speakerOut*self.numLearners+extra)
        else:
            self.linear0 = nn.Linear(self.speakerOut*self.numLearners,
                                     self.speakerOut)
            # self.norm0 = nn.BatchNorm1d(self.speakerOut*self.numLearners)

        self.linear1 = nn.Linear(self.speakerOut, int(self.speakerOut/2))
        self.classify = nn.Linear(int(self.speakerOut/2), 5)

    def forward(self, x):

        p = []
        for i in range(len(self.learners)):
            p.append(self.learners[i](x[:, i*self.speakerIn:(i+1)*self.speakerIn]))

        if self.extra != 0:
            p.append(x[:, self.speakerIn*self.numLearners:])

        x = torch.cat((p), dim=1)

        x = F.relu(self.linear0(F.dropout(x)))
        x = F.relu(self.linear1(x))
        x = self.classify(x)
        x = F.softmax(x, 1)

        return x


