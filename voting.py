#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:03:19 2023
voting.py
@author: richard
"""

from dataSets import generateData, trainTestValSplit
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier

#%% Generate data
standardData = True
augmented = True
augmentedSet = 2
projectPath = '/home/richard/documents/school/2022Fall/'
# projectPath = '/home/richard/Documents/gits/speakers/'
dataPath = projectPath + 'data/'
modelPath = projectPath + 'models/'
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

method = 'combination'
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
    # video = 'isaac'
    testVideos = ['/Luis_Liz_Clipped/']
    video = 'Luis_Liz_Clipped'
    with open(projectPath+'standardDataList.p', 'rb') as f:
        data = pickle.load(f)

    data = [projectPath+s for s in data]
    allData, allScores, allY, allDur = generateData(method=method, data=data)

data = []
if method == 'tony' or method == 'euclidean':
    data = allScores

elif method == 'none':
    for i in range(len(allData)):
        d = np.array(pd.read_csv(allData[i],delimiter=',',header=None)).flatten()
        data.append(d)

elif method == 'difference' or method == 'allMatrices':
    for s in allScores:
        data.append(np.array(s).flatten())

elif method == 'combination':
    for s in allScores:
        data.append(np.concatenate(s))

dataTuples = []
for i in range(len(allY)):
    dataTuples.append((data[i], allY[i], allData[i], allDur[i]))

totals = [0] * 5
for i in range(5):
    total = 0
    for j in allY:
        if j-1 == i:
            total += 1

    totals[i] = total

print('Total Samples:', str(sum(totals)))
for i in range(len(totals)):
    print('\tSpeaker '+str(i+1)+': '+str(totals[i]))
print('Total audio length:', sum(allDur))
print('Method:', method)

seed = 3748
train, test, val = trainTestValSplit(dataTuples, .85, .15, 0, 5, seed, testVideos)
trainX = []
trainDur = []
trainY = []
testX = []
testDur = []
testY = []
for i in range(len(train)):
    trainX.append(train[i][0])
    trainDur.append(train[i][3])
    trainY.append(train[i][1]-1)

for i in range(len(test)):
    testX.append(test[i][0])
    testDur.append(test[i][3])
    testY.append(test[i][1]-1)

#%% Load models
svm = glob.glob(modelPath+video+method[0].upper()+method[1:]+'Svm*')
knn = glob.glob(modelPath+video+method[0].upper()+method[1:]+'knn*')
forest = glob.glob(modelPath+video+method[0].upper()+method[1:]+'Forest*')
xgb = glob.glob(modelPath+video+method[0].upper()+method[1:]+'Xgb*')

with open(svm[0], 'rb') as file:
    svmLog = pickle.load(file)
with open(knn[0], 'rb') as file:
    knnLog = pickle.load(file)
with open(forest[0], 'rb') as file:
    forestLog = pickle.load(file)
with open(xgb[0], 'rb') as file:
    xgbLog = pickle.load(file)

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

svm = svmLog['model']
knn = knnLog['model']
forest = forestLog['model']
xgb = xgbLog['model']

classifier = VotingClassifier(estimators=[('svm', svm),
                                          ('knn', knn),
                                          ('forest', forest),
                                          ('xgb', xgb)],
                              voting='soft')

#%% Testing
classifier.fit(trainX, trainY)

predY = classifier.predict(testX)
correct = 0
wrongTime = 0
lengthsY = [0]*5
lengthsPred = [0]*5
for i in range(len(predY)):
    lengthsY[testY[i]] += testDur[i]
    lengthsPred[int(predY[i])] += testDur[i]
    if testY[i] == predY[i]:
        correct += 1
    else:
        wrongTime += testDur[i]

acc = correct/len(testY)
der = wrongTime/sum(lengthsY)
print('acc:', acc)
print('der:', der)
