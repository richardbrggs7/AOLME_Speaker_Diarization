#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:34:00 2022
test.py
@author: richard
"""
from dataSets import getDuration, generateData, trainTestValSplit
import math
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from speakerTransformerClassifier import speakerTransformerClassifier
import xgboost as xgb
from scipy.io.wavfile import read, write
from scipy import signal
import glob

#%% Accessing models
# method = 'none'
# projectPath = '/home/richard/documents/school/2022Fall/'
# testVideo = 'Luis_Liz_Clipped'
# # testVideo = 'isaac'
# models = glob.glob(projectPath + 'models/' + testVideo + method[0].upper()+method[1:] + '*.p' )

# print(method)
# print(testVideo)
# for m in models:

#     if m.find('Net') != -1:
#         with open(m, 'rb') as f:
#             netModelDict = pickle.load(f)
#         print('Net:')
#         print('\tAcc:   ', netModelDict['testAcc'])
#         print('\tDER:   ', netModelDict['testDer'])
#         print('\tLr:    ', netModelDict['lr'])

#     if m.find('Svm') != -1:
#         with open(m, 'rb') as f:
#             svmModelDict = pickle.load(f)
#         print('SVM:')
#         print('\tAcc:   ', svmModelDict['acc'])
#         print('\tDER:   ', svmModelDict['der'])
#         print('\tgamma: ', svmModelDict['gamma'])
#         print('\tC:     ', svmModelDict['c'])

#     elif m.find('knn') != -1:
#         with open(m, 'rb') as f:
#             knnModelDict = pickle.load(f)
#         print('Knn:')
#         print('\tAcc:   ', knnModelDict['acc'])
#         print('\tDER:   ', knnModelDict['der'])
#         print('\tn:     ', knnModelDict['n_neighbors'])

#     elif m.find('Forest') != -1:
#         with open(m, 'rb') as f:
#             forModelDict = pickle.load(f)
#         print('Forest:')
#         print('\tAcc:   ', forModelDict['acc'])
#         print('\tDER:   ', forModelDict['der'])
#         print('\test:   ', forModelDict['n_estimators'])
#         print('\tdepth: ', forModelDict['max_depth'])

#     elif m.find('Xgb') != -1:
#         with open(m, 'rb') as f:
#             xgbModelDict = pickle.load(f)
#         print('XGB:')
#         print('\tAcc:   ', xgbModelDict['acc'])
#         print('\tDER:   ', xgbModelDict['der'])
#         print('\test:   ', xgbModelDict['n_estimators'])
#         print('\tdepth: ', xgbModelDict['max_depth'])
#         print('\teta:   ', xgbModelDict['eta'])
#         print('\tgamma: ', xgbModelDict['gamma'])
#         print('\tlambda:', xgbModelDict['lambda'])

#%% Data statistics
standardData = False
augmented = False
augmentedSet = 2
method = 'allMatrices'
projectPath = '/home/richard/documents/school/2022Fall/'
# projectPath = '/home/richard/Documents/gits/speakers/'
dataPath = projectPath + 'data/'
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
    testVideos = None
    for s in datasets:
        data, scores, ys , dur = generateData(s, 0, method)
        allData += data
        allScores += scores
        allY += ys
        allDur += dur
else:
    testVideos = ['/LIZ_E/']
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
print('Average sample length:', sum(allDur)/len(allDur))
print('Method:', method)

durs = [[], [], [], [], []]
maxes = [-1] * 5
mins = [-1] * 5
sums = [-1] * 5
avgs = [-1] * 5
devs = [-1] * 5

for i in range(5):
    dur = []
    for d in dataTuples:
        if d[1]-1 == i:
            dur.append(d[3])

    durs[i] = dur

for i in range(5):
    maxes[i] = max(durs[i])
    mins[i] = min(durs[i])
    sums[i] = sum(durs[i])
    avgs[i] = sums[i] / len(durs[i])
    dev = 0
    for j in durs[i]:
        dev += (j - avgs[i])**2

    dev = math.sqrt(dev/len(durs[i]))
    devs[i] = dev

videoTimes = [0] * len(datasets)
for i, s in enumerate(datasets):
    for d in dataTuples:
        if d[2].find(s) != -1:
            videoTimes[i] += d[3]

print('Sums:\t', sums)
print('Maxes:\t', maxes)
print('Mins:\t', mins)
print('Avgs:\t', avgs)
print('Devs:\t', devs)
for i, s in enumerate(datasets):
    print(s+' length: '+str(videoTimes[i]))

#%% Tree features
# method = 'none'
# video = 'Luis_Liz_Clipped'
# modelPath = '/home/richard/documents/school/2022Fall/models/'
# with open(modelPath+video+method[0].upper()+method[1:]+'Xgb.p', 'rb') as f:
#     xgbDict = pickle.load(f)
# model = xgbDict['model']

# # print(model.__dir__())
# # print()
# # print(model.get_params())
# # print()
# # print(model.get_xgb_params())

# model._Booster.dump_model(modelPath+'test.json', dump_format='json')

# with open(modelPath+'test.json', 'rb') as f:
#     lines = f.readlines()

# xgbCount = 0
# for l in lines:
#     if str(l).find('nodeid') != -1:
#         xgbCount += 1

# with open(modelPath+video+method[0].upper()+method[1:]+'Forest.p', 'rb') as f:
#     forestDict = pickle.load(f)
# model = forestDict['model']

# forestCount = 0
# for t in model.estimators_:
#     forestCount += t.tree_.node_count

# print(method+' Forest param count:\t\t\t\t', forestCount)

# print(method+' XGB param count:\t\t\t\t\t', xgbCount)

# print(method+' Voting Classifier param count:\t', xgbCount+forestCount)


#%% XGB importance
# with open('/home/richard/documents/school/2022Fall/models/allMatricesXgb.p', 'rb') as f:
#     xgbDict = pickle.load(f)

# model = xgbDict['model']
# i = model.feature_importances_
# matrices = []
# for x in range(6):
#     matrices.append(np.reshape(i[x*105:(x+1)*105], (5, -1)))

# yLabels = ['S1', 'S2', 'S3', 'S4', 'S5']
# colormap = 'hot'

# fig, axs = plt.subplots(6, 1, dpi=300, gridspec_kw={'hspace':.4})

# for i, ax in enumerate(axs):
#     if i == 0:
#         ax.set_title('Test Sample', fontsize=9, pad=1)
#     else:
#         ax.set_title('Reference Matrix '+str(i), fontsize=9, pad=1)
#     ax.imshow(matrices[i], cmap=colormap)
#     ax.set_yticks(np.arange(len(yLabels)), labels=yLabels, fontsize=6)
#     ax.set_xticks([])

# bar = fig.colorbar(ScalarMappable(cmap=colormap), ax=axs[:], shrink=.8, pad=.03, aspect=30)
# bar.ax.tick_params(labelsize=8)

#%% XGB param testing
# standardData = True
# augmented = True
# augmentedSet = 2
# projectPath = '/home/richard/documents/school/2022Fall/'
# # dataPath = '/home/richard/documents/school/2022Fall/data'
# # picklePath = '/home/richard/Documents/gits/speakers/'
# dataPath = projectPath + 'data/'
# if not standardData:
#     datasets = ['/LIZ_E/', '/Ivone_E_geo2/', '/LIZ_E0205/', '/LIZ_E0305/',
#                 '/Luis_Liz_Clipped/',
#                 '/Isaac_C2L1W_q2_02-04_part1/', '/Isaac_C2L1W_q2_02-04_part2/',
#                 '/G-C1L1P-Mar02-E-Irma_q2_03-08/', '/G-C1L1P-Mar02-E-Irma_q2_05-08/'
#                 ]
#     if augmented:
#         datasets = datasets + ['/AUG'+str(augmentedSet)+'/'+d.replace('/','')+'_AUG/' for d in datasets]

#     datasets = [dataPath + d for d in datasets]
#     datasets = [d + 'CCMatrices/' for d in datasets]

# method = 'combination'
# dataPaths = []
# allData = []
# allScores = []
# allY = []
# allDur = []
# if not standardData:
#     for s in datasets:
#         data, scores, ys , dur = generateData(s, 0, method)
#         allData += data
#         allScores += scores
#         allY += ys
#         allDur += dur
# else:
#     with open(projectPath+'standardDataList.p', 'rb') as f:
#         data = pickle.load(f)

#     data = [projectPath+s for s in data]
#     allData, allScores, allY, allDur = generateData(method=method, data=data)

# data = []
# if method == 'tony' or method == 'euclidean':
#     data = allScores

# elif method == 'none':
#     for i in range(len(allData)):
#         d = np.array(pd.read_csv(allData[i],delimiter=',',header=None)).flatten()
#         data.append(d)

# elif method == 'difference' or method == 'allMatrices':
#     for s in allScores:
#         data.append(np.array(s).flatten())

# elif method == 'combination':
#     for s in allScores:
#         data.append(np.concatenate(s))

# dataTuples = []
# for i in range(len(allY)):
#     dataTuples.append((data[i], allY[i], allData[i], allDur[i]))

# totals = [0] * 5
# for i in range(5):
#     total = 0
#     for j in allY:
#         if j-1 == i:
#             total += 1

#     totals[i] = total

# print('Total Samples:', str(sum(totals)))
# for i in range(len(totals)):
#     print('\tSpeaker '+str(i+1)+': '+str(totals[i]))
# print('Total audio length:', sum(allDur))
# print('Method:', method)

# seed = 3748
# train, test, val = trainTestValSplit(dataTuples, .85, .15, 0, 5, seed)
# trainX = []
# trainDur = []
# trainY = []
# testX = []
# testDur = []
# testY = []
# for i in range(len(train)):
#     trainX.append(train[i][0])
#     trainDur.append(train[i][3])
#     trainY.append(train[i][1]-1)

# for i in range(len(test)):
#     testX.append(test[i][0])
#     testDur.append(test[i][3])
#     testY.append(test[i][1]-1)

# model=xgb.XGBClassifier(eval_metric='mlogloss',use_label_encoder=False,reg_lambda=5, alpha=1, eta=1)
# model.fit(trainX, trainY)

# print(model.get_params())
# print(model.get_xgb_params())
# # print(model.min_split_loss)
# print(model.reg_alpha)
# print(model.get_params()['reg_alpha'])
# print(model.get_params()['learning_rate'])
# print(model.get_params()['reg_lambda'])

#%% Losses graphs
# with open('/home/richard/Documents/gits/speakers/models/combinationNet.p', 'rb') as f:
#     log = pickle.load(f)

# plt.figure(figsize=(7,5))
# # plt.title('Loss over Training', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Loss', fontsize=16)
# plt.plot(log['trainLoss'], label='Train')
# plt.plot(log['valLoss'], label='Validation')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14,rotation=60)
# plt.legend(fontsize=12)
# plt.show()

#%% Spectrograms
# dataDir = '/home/richard/Documents/gits/speakers/data/'
# # file = 'G-C2L1P-Feb16-B-Shelby_q2_03-05_duplicate_of_Luis_Liz_Clipped/G-C2L1P-Feb16-B-Shelby_q2_03-05_001_Emily62P_0005_S5.wav'
# data = 'LIZ_E'
# file = '/S3_1.wav'
# aolmeRate, aolme = read(dataDir+data+file)
# # homeRate, home = read('/home/richard/Downloads/callHomeSample.wav')

# plt.figure(dpi=200)
# plt.specgram(aolme, Fs=aolmeRate, cmap="inferno")
# # plt.title('Spectrogram of AOLME Sample')
# plt.title('Original')
# plt.xlabel("Time (s)", fontsize=18)
# plt.ylabel("Frequency (Hz)", fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14,rotation=60)
# plt.show()

# aolmeRate, aolme = read(dataDir+'AUG0/'+data+'_AUG/'+file)
# plt.figure(dpi=200)
# plt.specgram(aolme, Fs=aolmeRate, cmap="inferno")
# # plt.title('Spectrogram of AOLME Sample')
# plt.title('AUG0')
# plt.xlabel("Time (s)", fontsize=18)
# plt.ylabel("Frequency (Hz)", fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14,rotation=60)
# plt.show()

# aolmeRate, aolme = read(dataDir+'AUG1/'+data+'_AUG/'+file)
# plt.figure(dpi=200)
# plt.specgram(aolme, Fs=aolmeRate, cmap="inferno")
# # plt.title('Spectrogram of AOLME Sample')
# plt.title('AUG1')
# plt.xlabel("Time (s)", fontsize=18)
# plt.ylabel("Frequency (Hz)", fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14,rotation=60)
# plt.show()

# aolmeRate, aolme = read(dataDir+'AUG2/'+data+'_AUG/'+file)
# plt.figure(dpi=200)
# plt.specgram(aolme, Fs=aolmeRate, cmap="inferno")
# # plt.title('Spectrogram of AOLME Sample')
# plt.title('AUG2')
# plt.xlabel("Time (s)", fontsize=18)
# plt.ylabel("Frequency (Hz)", fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14,rotation=60)
# plt.show()

# # plt.specgram(home[13000:43000], Fs=homeRate, cmap="inferno")
# # plt.title('Spectrogram of CALLHOME Sample')
# # plt.xlabel("Time (s)")
# # plt.ylabel("Frequency (Hz)")
# # plt.show()

# # frequencies, times, spectrogram = signal.spectrogram(aolme, aolmeRate)

# # plt.pcolormesh(times, frequencies, spectrogram)
# # plt.imshow(spectrogram)
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec]')
# # plt.show()

# # frequencies, times, spectrogram = signal.spectrogram(home[:10000], homeRate)

# # plt.pcolormesh(times, frequencies, spectrogram)
# # plt.imshow(spectrogram)
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec]')
# # plt.show()
