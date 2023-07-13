#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:17:49 2023
speakerTransformerClassifier.py
@author: richard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class speakerTransformerClassifier(nn.Module):
    def __init__(self, numEncoders, encoderIn, nHeads, dimFC, device, extra=0):
        """
        Initialize the transformer-using speaker classifier

        Parameters
        ----------
        numEncoders : int
            Number of encoders to use. Should be equal to the number of
            flattened matrices output by the method chosen in generateData.
        encoderIn : int
            Input dimension of encoders. Should generally be 105, since that is
            the size of each matrix.
        nHeads : int
            Number of heads of each encoder.
        dimFC : int
            Dimensions of fully connected layers of the encoders.
        device : str
            Device used by pytorch (cpu or cuda:#).
        extra : int, optional
            Extra numbers a generateData method outputs not connected to a
            matrix (i.e. the tony method numbers attached to the end of the
                    combination method). The default is 0.

        Returns
        -------
        None.

        """

        super(speakerTransformerClassifier, self).__init__()
        self.encoderIn = encoderIn
        self.device = device
        self.numEncoders = numEncoders
        self.encoders = nn.ParameterList()
        self.extra = extra
        for i in range(self.numEncoders):
            self.encoders.append(nn.TransformerEncoderLayer(encoderIn, nHeads,
                                                            dim_feedforward=dimFC,
                                                            batch_first=True).to(device,dtype=torch.double))

        if extra != 0:
            self.linear0 = nn.Linear(self.encoderIn*self.numEncoders+extra, self.encoderIn)
            self.norm0 = nn.BatchNorm1d(self.encoderIn*self.numEncoders+extra)
        else:
            self.linear0 = nn.Linear(self.encoderIn*self.numEncoders, self.encoderIn)
            self.norm0 = nn.BatchNorm1d(self.encoderIn*self.numEncoders)

        self.linear1 = nn.Linear(self.encoderIn, int(self.encoderIn/2))
        self.predict = nn.Linear(int(self.encoderIn/2), 5)


    def forward(self, x):
        p = []
        for i in range(len(self.encoders)):
            p.append(self.encoders[i](x[:, i*self.encoderIn:(i+1)*self.encoderIn]))

        if self.extra != 0:
            p.append(x[:, self.encoderIn*self.numEncoders:])

        # for i in p:
        #     print(i.size())

        p = torch.cat((p), dim=1)
        # p = torch.cat((x1, x2, x3, x4, x5), dim=1)
        p = F.relu(self.linear0(self.norm0(p)))
        p = F.relu(self.linear1(F.dropout(p)))
        p = self.predict(p)
        p = F.softmax(p, 1)

        return p
