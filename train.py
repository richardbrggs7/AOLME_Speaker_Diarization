#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:17:49 2023
testTransformer.py
@author: richard
"""

from dataSets import generateData, CCMatrixData, trainTestValSplit
import pickle
from speakerClassifier import speakerClassifier
from speakerTransformerClassifier import speakerTransformerClassifier
from linearClassifier import linearClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trainAndTest import trainLoop, testLoop

#%% Hyperparams
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

randomSeed = 3748
projectPath = '/home/richard/documents/school/2022Fall/'
# projectPath = '/home/richard/Documents/gits/speakers/'
epochs = 200
batchSize = 100
stoppingBuffer = 20     # After this amount of epochs, early stopping will be considered
opt = 'adam'
lr = .01
decay = 1e-4
momentum = 0
numWorkers = 4          # Number of workers for DataLoader
netType = 'speakerLearner'
standardData = True
augmented = True
augmentedSet = 2
method = 'combination' # The data input method used in generateData
extra = 0

learnerMid = 40
learnerOut = 20

nHeads = 5
dimFc = 30

if method == 'none':
    numEncoders = 1
    encoderIn = 105
elif method == 'allMatrices':
    numEncoders = 6
    encoderIn = 105
elif method == 'difference':
    numEncoders = 5
    encoderIn = 105
elif method == 'combination':
    numEncoders = 11
    encoderIn = 105
    extra = 5

#%% Prepare data
# dataPath = projectPath+'/data'
dataPath = projectPath+'/data'
if not standardData:
    datasets = ['/LIZ_E/', '/Ivone_E_geo2/', '/LIZ_E0205/', '/LIZ_E0305/',
                '/Luis_Liz_Clipped/',
                '/Isaac_C2L1W_q2_02-04_part1/', '/Isaac_C2L1W_q2_02-04_part2/',
                '/G-C1L1P-Mar02-E-Irma_q2_03-08/', '/G-C1L1P-Mar02-E-Irma_q2_05-08/'
                ]
    if augmented:
        datasets = datasets + ['/AUG'+str(augmentedSet)+'/'+d.replace('/','')+'_AUG/' for d in datasets]

    datasets = [dataPath + d for d in datasets]
    datasets = [d + 'CCMatrices/' for d in datasets]

dataPaths = []
allData = []
allScores = []
allY = []
allDur = []
if not standardData:
    for s in datasets:
        data, scores, ys , dur = generateData(s, 0, method)
        allData += data
        allScores += scores
        allY += ys
        allDur += dur
else:
    # testVideos = ['/Isaac_C2L1W_q2_02-04_part1/', '/Isaac_C2L1W_q2_02-04_part2/']
    testVideos = ['/Luis_Liz_Clipped/']
    with open(projectPath+'standardDataList.p', 'rb') as f:
        data = pickle.load(f)

    data = [projectPath+s for s in data]
    allData, allScores, allY, allDur = generateData(method=method, data=data)

dataTuples = []
for i in range(len(allScores)):
    dataTuples.append((allScores[i], allY[i], allData[i], allDur[i]))

train, test, val = trainTestValSplit(dataTuples, .85, .15, .15, 5, seed=randomSeed, testVideos=testVideos)
x = []
y = []
dur = []
for X, Y, _, D in train:
    x.append(X)
    y.append(Y)
    dur.append(D)
train = CCMatrixData(x, y, dur)

x = []
y = []
dur = []
for X, Y, _, D in test:
    x.append(X)
    y.append(Y)
    dur.append(D)
test = CCMatrixData(x, y, dur)

x = []
y = []
dur = []
for X, Y, _, D in val:
    x.append(X)
    y.append(Y)
    dur.append(D)
val = CCMatrixData(x, y, dur)

trainLoader = DataLoader(train, batch_size=batchSize, shuffle=True,
                          num_workers=numWorkers)
testLoader = DataLoader(test, batch_size=batchSize, shuffle=True,
                        num_workers=numWorkers)
valLoader = DataLoader(val, batch_size=batchSize, shuffle=True,
                        num_workers=numWorkers)

print('Train:', len(train))
print('Test:', len(test))
print('Val:', len(val))

#%% Net, Loss and Optimizer
if netType == 'transformer':
    net = speakerTransformerClassifier(numEncoders,encoderIn,nHeads,dimFc,device,
                                        extra=extra).to(device=device, dtype=torch.double)
elif netType == 'speakerLearner':
    net = speakerClassifier(numEncoders,encoderIn,learnerMid,learnerOut,
                                        extra=extra).to(device=device, dtype=torch.double)
elif netType == 'linear':
    net = linearClassifier(numEncoders*encoderIn+extra, 30).to(device=device, dtype=torch.double)

if opt == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
elif opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=decay, momentum=momentum)
elif opt =='adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=lr, weight_decay=decay)
elif opt == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=decay)

lossF = nn.CrossEntropyLoss()
params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Total parameters:', params)

#%% Training
results = trainLoop(net, epochs, trainLoader, valLoader, optimizer, lossF,
                    stoppingBuffer, device)
results['optimizer'] = opt
results['lr'] = lr
results['decay'] = decay
results['momentum'] = momentum
results['method'] = method
results['netType'] = netType
results['seed'] = randomSeed

if netType == 'speakerLearner':
    results['learnerMid'] = learnerMid
    results['learnerOut'] = learnerOut
elif netType == 'transformer':
    results['nHeads'] = nHeads
    results['dimFc'] = dimFc

#%% Testing
net.load_state_dict(results['net'])
net.to(device)
acc, der, _ = testLoop(net, testLoader, device)

results['testAcc'] = acc
results['testDer'] = der
results['testVideos'] = testVideos
if augmented:
    results['augmentedSet'] = augmentedSet
else:
    results['augmentedSet'] = None

with open(projectPath+method+'Net.p', 'wb') as f:
    pickle.dump(results, f)
