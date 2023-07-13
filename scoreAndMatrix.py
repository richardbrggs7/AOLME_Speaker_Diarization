#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 13:24:24 2023
scoreAndMatrix.py
@author: richard
"""

from dataSets import generateData, trainTestValSplit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from pushbullet import Pushbullet

#%% Generate data
standardData = True
augmented = True
augmentedSet = 2
method = 'none'
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
#    testVideos = ['/Isaac_C2L1W_q2_02-04_part1/', '/Isaac_C2L1W_q2_02-04_part2/']
    testVideos = ['/Luis_Liz_Clipped/']
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

#%% If using Tony's method, check accuracy without machine learning
if method == 'tony' or method == 'euclidean':
    correct = 0
    durY = [0] * 5
    durPred = [0] * 5
    predY = []
    wrongPred = 0
    for i in range(len(allY)):
        if method == 'tony':
            ans = np.argmax(allScores[i])
        else:
            ans = np.argmin(allScores[i])

        predY.append(ans)
        if ans == allY[i]:
            correct += 1
        else:
            wrongPred += allDur[i]

        durPred[ans] += allDur[i]
        durY[allY[i]] += allDur[i]

    acc = correct / len(allY)
    der = wrongPred / sum(allDur)

    print(20*'-'+'Max Prediction'+'-'*20)
    print('Accuracy: {:.3f}'.format(acc))
    print('DER: {:.3f}'.format(der))
    print('Error rate per speaker:')
    for i in range(len(durY)):
        error = abs(durY[i] - durPred[i])/durY[i]
        print('\tSpeaker'+str(i+1)+': {:.3f}'.format(error))

    con = confusion_matrix(allY, predY)
    disp = ConfusionMatrixDisplay(con, display_labels=[1,2,3,4,5])
    disp.plot()
    disp.ax_.set_title('Distances, Max, Acc: {:.3f}'.format(acc))
    plt.show()

#%% Fit models and test
names = [
            'SVM',
            'KNN',
            'Random Forest',
            'XGBoost',
        ]

models = []
if names.count('SVM') != 0:
    models.append(GridSearchCV(SVC(probability=True), {'C' : [.1, .33, .75, 1, 5, 15, 30, 60],
                                                        'gamma' : ['scale', 'auto']
                                                        },
                                scoring='accuracy',verbose=2))

if names.count('KNN') != 0:
    models.append(GridSearchCV(KNeighborsClassifier(n_jobs=-1),
                                {'n_neighbors' : [3, 4, 5, 6, 8, 15, 30]},
                                scoring='accuracy',verbose=2))

if names.count('Random Forest') != 0:
    models.append(GridSearchCV(RandomForestClassifier(n_jobs=-1),
                                {'n_estimators' : [50, 100, 150, 200, 250],
                                'max_depth' : [None, 200, 150, 100, 50, 20]
                                },
                                scoring='accuracy',verbose=2))

if names.count('XGBoost') != 0:
    models.append(GridSearchCV(xgb.XGBClassifier(n_jobs=-1,eval_metric='mlogloss',use_label_encoder=False),
                                {'max_depth' : [None, 4, 6, 10, 50, 100],
                                'n_estimators' : [50, 150, 250, 500, 1000],
                                'eta' : [0.1, 0.3, 0.5],
                                'gamma' : [0, 3, 6],
                                'lambda' : [0, 1, 3, 6]
                                },
                                scoring='accuracy',verbose=2))

for name, model in zip(names, models):
    # print(20*'-'+name+'-'*20)
    estimators = model.fit(trainX, trainY)
    est = estimators.best_estimator_
    predY = est.predict(testX)

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
    if name == 'XGBoost':
        xgb = est
        xgAcc = acc
        xgDer = der
    elif name == 'SVM':
        svm = est
        svmAcc = acc
        svmDer = der
    elif name=='KNN':
        knn = est
        kAcc = acc
        kDer = der
    elif name=='Random Forest':
        forest = est
        forAcc = acc
        forDer = der

# %% Print Results
print('Seed: '+str(seed))
if names.count('SVM') != 0:
    print('SVM:')
    print('\tC: '+str(svm.C))
    print('\tgamma: '+str(svm.gamma))
    print('\tAcc: '+str(svmAcc))
    print('\tDER: '+str(svmDer))

if names.count('KNN') != 0:
    print('KNN:')
    print('\tn_neighbors: '+str(knn.n_neighbors))
    print('\tAcc: '+str(kAcc))
    print('\tDER: '+str(kDer))

if names.count('Random Forest') != 0:
    print('Forest:')
    print('\tn_estimators: '+str(forest.n_estimators))
    print('\tmax_depth: '+str(forest.max_depth))
    # print('\tparams: '+str(sum(tree.tree_.node_count for tree in forest.estimators_) * 5))
    print('\tAcc: '+str(forAcc))
    print('\tDER: '+str(forDer))

if names.count('XGBoost') != 0:
    print('XGB:')
    print('\tn_estimators: '+str(xgb.n_estimators))
    print('\tmax_depth: '+str(xgb.max_depth))
    if xgb.max_depth == None:
        print('\t\tdepth: '+str(xgb.get_xgb_params()['max_depth']))
    print('\teta: '+str(xgb.get_params()['learning_rate']))
    print('\tgamma: '+str(xgb.get_params()['gamma']))
    print('\tlambda: '+str(xgb.get_params()['reg_lambda']))
    # print('\talpha: '+str(xgb.get_params()['reg_alpha']))
    # print('\tparams: '+str(len(xgb._Booster.trees_to_dataframe())))
    print('\tAcc: '+str(xgAcc))
    print('\tDER: '+str(xgDer))

# %% Save Results
if names.count('SVM') != 0:
    svmDict = {}
    svmDict['seed'] = seed
    svmDict['augmentedSet'] = augmentedSet
    svmDict['acc'] = svmAcc
    svmDict['der'] = svmDer
    svmDict['method'] = method
    svmDict['c'] = svm.C
    svmDict['gamma'] = svm.gamma
    svmDict['model'] = svm
    with open(projectPath+method+'Svm.p', 'wb') as file:
        pickle.dump(svmDict, file)

if names.count('KNN') != 0:
    knnDict = {}
    knnDict['seed'] = seed
    knnDict['augmentedSet'] = augmentedSet
    knnDict['acc'] = kAcc
    knnDict['der'] = kDer
    knnDict['method'] = method
    knnDict['n_neighbors'] = knn.n_neighbors
    knnDict['model'] = knn
    with open(projectPath+method+'knn.p', 'wb') as file:
        pickle.dump(knnDict, file)

if names.count('Random Forest') != 0:
    forestDict = {}
    forestDict['seed'] = seed
    forestDict['augmentedSet'] = augmentedSet
    forestDict['acc'] = forAcc
    forestDict['der'] = forDer
    forestDict['method'] = method
    forestDict['n_estimators'] = forest.n_estimators
    forestDict['max_depth'] = forest.max_depth
    forestDict['params'] = sum(tree.tree_.node_count for tree in forest.estimators_) * 5
    forestDict['model'] = forest
    with open(projectPath+method+'Forest.p', 'wb') as file:
        pickle.dump(forestDict, file)

if names.count('XGBoost') != 0:
    xgbDict = {}
    xgbDict['seed'] = seed
    xgbDict['augmentedSet'] = augmentedSet
    xgbDict['acc'] = xgAcc
    xgbDict['der'] = xgDer
    xgbDict['method'] = method
    xgbDict['n_estimators'] = xgb.n_estimators
    xgbDict['max_depth'] = xgb.max_depth
    xgbDict['depth'] = xgb.get_xgb_params()['max_depth']
    # xgbDict['params'] = len(xgb._Booster.trees_to_dataframe())
    xgbDict['eta'] = xgb.get_params()['learning_rate']
    xgbDict['gamma'] = xgb.get_params()['gamma']
    xgbDict['lambda'] = xgb.get_params()['reg_lambda']
    # xgbDict['alpha'] = xgb.get_params()['reg_alpha']
    xgbDict['model'] = xgb
    with open(projectPath+method+'Xgb.p', 'wb') as file:
        pickle.dump(xgbDict, file)

pb = Pushbullet('o.kdgOlrPr8lmWiYJpdb91sVeEWXbyMeBp')
push = pb.push_note('Done', 'Validation is finished at Spacer')
