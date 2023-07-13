#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:55:17 2022
dataSets.py
@author: spacer
"""

import glob
import math
import numpy as np
import os
import pandas as pd
import random
from scipy.io.wavfile import read
import torch
from torch.utils.data import Dataset

class CCMatrixData(Dataset):
    """
    Generates the dataset used for speakerClassifier

    Parameters
    ----------
    ccList : list
        List of paths to the CCMatrices of data

    Returns
    -------
    Dataset

    """
    def __init__(self, data, targets, durations, numClasses=5):
        """
        Initialize Dataset

        Parameters
        ----------
        data : list
            List of all items in the dtatset.
        targets : list
            List of all targets in the dataset
        durations : list
            List of durations of each item in the dataset
        numClasses : int
            Number of classes in the dataset.

        Returns
        -------
        Dataset.

        """

        self.data = data
        self.targets = targets
        self.durations = durations
        self.classes = numClasses

        # If data is not flat, flatten it now so it is not necessary in getitem
        if type(data[0]) == type(list()) or len(list(data[0].shape)) != 1:
            tempData = []
            for d in data:
                temp = []
                for mat in d:
                    array = mat.flatten()
                    temp += list(array)

                tempData.append(np.array(temp))

            self.data = tempData

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i]), self.targets[i]-1, self.durations[i]

def trainTestValSplit(data, trainPercent, testPercent, valPercent,
                      numClasses, seed=0, testVideos=None):
    """
    Splits data into stratified subsets

    Parameters
    ----------
    data : list of tuples
        List of data points as tuples; should be of the shape
        [data-to-be-shuffled, targets, file-names, anyting-else...].
    trainPercent : float
        Percentage of data assigned to train.
    testPercent : float
        Percentage of data assigned to test.
    valPercent : float
        Percentage of data assigned to val.
    numClasses : int
        Number of classes in the dataset.
    seed : int
        Seed for randomizing data shuffle. Default is 0 (no shuffle).
    testVideos : list of strings
        If not None, the videos the test set will be taken from. Should be a
        subset of the total data set

    Returns
    -------
    train : list
        List of data points.
    test : list
        List of data points.
    val : list
        List of data points.

    """
    train = []
    test = []
    val = []
    if testVideos == None:
        assert trainPercent + testPercent + valPercent == 1
        if seed != 0:
            random.seed(seed)
            random.shuffle(data)

        for c in range(numClasses):
            thisClass = []
            for i in range(len(data)):
                # Ensure unaugmented data is first in the list so it is not in the
                # test subset
                if data[i][1]-1 == c and data[i][2].find('AUG') == -1:
                    thisClass.append(data[i])
            for i in range(len(data)):
                # Add augmented data at the end
                if data[i][1]-1 == c and data[i][2].find('AUG') != -1:
                    thisClass.append(data[i])

            testTrainIndex = math.floor(len(thisClass)*testPercent)
            trainValIndex = math.floor(testTrainIndex + len(thisClass)*trainPercent)
            test += thisClass[:testTrainIndex]
            if valPercent != 0:
                train += thisClass[testTrainIndex:trainValIndex+1]
                val += thisClass[trainValIndex+1:]
            else:
                train += thisClass[testTrainIndex:]

    # If a validation set is needed
    elif valPercent != 0:
        # Find unaugmented data points belonging to the videos and add
        # them to test
        newData = []
        for i, d in enumerate(data):
            isTest = 0
            for video in testVideos:
                if d[2].find(video) != -1 and d[2].find('AUG') == -1:
                    # test.append(data[i])
                    isTest += 1
                # else:
                    # newData.append(data[i])
            if isTest:
                test.append(data[i])
            else:
                newData.append(data[i])

        data = newData
        assert trainPercent + valPercent == 1
        if seed != 0:
            random.seed(seed)
            random.shuffle(data)

        for c in range(numClasses):
            thisClass = []
            for i in range(len(data)):
                if data[i][1]-1 == c:
                    thisClass.append(data[i])

            trainValIndex = math.floor(len(thisClass)*trainPercent)
            train += thisClass[:trainValIndex]
            val += thisClass[trainValIndex:]

    else:
        if seed != 0:
            random.seed(seed)
            random.shuffle(data)

        # Find unaugmented data points belonging to the videos and add
        # them to test
        for i, d in enumerate(data):
            isTest = 0
            for video in testVideos:
                if d[2].find(video) != -1 and d[2].find('AUG') == -1:
                    # test.append(data[i])
                    isTest += 1
                # else:
                    # train.append(data[i])
            if isTest:
                test.append(data[i])
            else:
                train.append(data[i])

    return train, test, val

def getDuration(path):
    """
    Get audio duration

    Parameters
    ----------
    path : str
        path to wav or csv file. If csv, the wav is expected to be one
        directory up, with the same name as the selected file.

    Returns
    -------
    Seconds duration of the file.

    """
    if path.endswith('.csv'):
        name = path.split('/')[-1]
        wavName = name.replace('.csv', '.wav')
        path = path.split('CCMatrices')[0]
        path = path + wavName
        sampleRate, data = read(path)
        dur = data.shape[0] / sampleRate
        return dur

    elif path.endswith('.wav'):
        sampleRate, data = read(path)
        dur = data.shape[0] / sampleRate
        return dur

def scorePoint(data, testList, threshold=1, method='tony'):
    """
    Scores a data point based on Tony's original approach, or other options;
    for Tony's method, for every index in the data array, if it is equal to
    that same index of a test array within a threshold for speaker x, speaker
    x gains 1 point.

    Parameters
    ----------
    data : numpy array
        Array for the data sample to be scored.
    testList : list of numpy arrays
        List of test arrays for each speaker in the dataset to which data
        belongs.
    threshold : int
        Threshold for determining whether an index is equal to that of the
        test array.
    method : str
        Method for determining distance. Options are "tony"'s original method
        "euclidean", "difference" - returns 5 vectors representing the
        difference between the sample and the test vectors - "allMatrices" -
        returns the sample and the 5 reference vectors - or "combination," which
        throws everything at the problem, returning the sample, test matrices,
        differences, and tony's method scores. Default is tony.

    Returns
    -------
    scores : list of ints
        Score for each speaker for the given data sample.

    """

    scores = []
    if method == 'tony':
        sample = data.flatten()
        for q in range(len(testList)):
            score = 0
            test = testList[q].flatten()
            for i in range(len(test)):
                if abs(sample[i] - test[i]) <= threshold:
                    score += 1

            scores.append(score)

    elif method == 'euclidean':
        for q in range(len(testList)):
            score = abs(np.linalg.norm(data - testList[q]))
            scores.append(score)

    elif method == 'difference':
        for q in range(len(testList)):
            score = np.abs(data - testList[q]).flatten()
            scores.append(score)

    elif method == 'allMatrices':
        scores.append(data.flatten())
        for q in range(len(testList)):
            scores.append(testList[q].flatten())

    elif method == 'combination':
        sample = data.flatten()
        scores.append(sample)
        for q in range(len(testList)):
            scores.append(testList[q].flatten())
        for q in range(len(testList)):
            score = np.abs(data - testList[q]).flatten()
            scores.append(score)
        temp = []
        for q in range(len(testList)):
            score = 0
            test = testList[q].flatten()
            for i in range(len(test)):
                if abs(sample[i] - test[i]) <= threshold:
                    score += 1

            temp.append(score)
        scores.append(np.array(temp, dtype=np.float64))

    return scores

def generateData(ccDir='', threshold=1, method='none', data=[]):
    """
    Generates data as a list to prepare for a PyTorch dataset, scoring a
    dataset based on various approaches. Since each dataset requires slightly
    different geometries, each will need its own set of testSources for
    comparison. There should be 1 test source for each speaker located in the
    CCMatrices/testSources folder of the dataset.

    Parameters
    ----------
    ccDir : path
        Path to the CCMatrices folder of the dataset to be tested. May be empty
        if data is not.
    threshold : int
        Threshold for testing whether an index is equal between the arrays to
        be scored and the testing arrays.
    method : str
        Method for determining distance. Options are "none", which simply
        returns the list of paths to the csvs, "tony"'s original method,
        "euclidean", "difference" - returns 5 vectors representing the
        difference between the sample and the test vectors - "allMatrices -
        returns the sample and the 5 reference vectors - or "combination, which
        throws everything and the kitchen sink at the problem, returning the
        sample, test matrices, differences, and tony's method scores. Default
        is tony.
    data : list of paths
        If empty, uses glob to generate a list of paths to the data points in
        each dataset; on a given machine, the order of the list is always the
        same, but on different machines glob generates the items in a different
        order. If not empty, must be a pre-made list of file names for data
        points to be generated.

    Returns
    -------
    data : CCMatrixData
        Dataset that was scored in the format from dataSets.py.
    scoresList : list
        For each data point, a list giving the score of each speaker; in
        theory, the index of the max is the speaker.
    answers : list
        For each data point, the actual speaker.
    durations : list
        For each datapoint, the time duration of that audio file.

    """
    if len(data) == 0:
        path = ccDir + '/*S1*'
        data = glob.glob(path)
        path = ccDir + '/*S2*'
        data = data + glob.glob(path)
        path = ccDir + '/*S3*'
        data = data + glob.glob(path)
        path = ccDir + '/*S4*'
        data = data + glob.glob(path)
        path = ccDir + '/*S5*'
        data = data + glob.glob(path)

    dataTuples = []
    for d in data:
        x = pd.read_csv(d, delimiter=',', header=None)
        x = torch.tensor(np.array(x))

        y = d.split('S')
        y = y[-1]
        y = int(y[0]) - 1    # Must be indexed from 0 not 1

        name = d.split('/')[-1]
        wavName = name.replace('.csv', '.wav')
        path = d.split('CCMatrices')[0]
        path = path + wavName
        sampleRate, temp = read(path)
        dur = temp.shape[0] / sampleRate

        dataTuples.append((x, y, dur, d))

    if len(ccDir) != 0:
        testData = ccDir + 'testSources/'
        testList = []
        for i in range(5):
            # speaker = glob.glob(testData+'S'+str(i+1)+'*')[0]
            # print(glob.glob(testData+'*S'+str(i+1)+'*'))
            for file in os.listdir(testData):
                if file.rfind('S'+str(i+1)) != -1:
                    # testList.append(file)
                    testList.append(np.array(pd.read_csv(testData+file,delimiter=',',header=None)))
                    # print(file)

            # testList.append(np.array(pd.read_csv(speaker,delimiter=',',header=None)))

        scoresList = []
        answers = []
        durations = []
        if method != 'none':
            for point, y, dur, _ in dataTuples:
                scoresList.append(scorePoint(np.array(point), testList, threshold, method))
                answers.append(y+1)
                durations.append(dur)
        else:
            for point, y, dur, _ in dataTuples:
                scoresList.append(np.array(point))
                answers.append(y+1)
                durations.append(dur)

        return data, scoresList, answers, durations

    else:
        scoresList = []
        answers = []
        durations = []
        if method != 'none':
            oldCCDir = ''
            for point, y, dur, path in dataTuples:
                # print('-'*50)
                # print(path)
                ccDir = path.split('CCMatrices')[0]
                # print(ccDir)
                if ccDir != oldCCDir:
                    testData = ccDir+'CCMatrices/testSources/'
                    testList = []
                    for i in range(5):
                        for file in os.listdir(testData):
                            if file.rfind('S'+str(i+1)) != -1:
                                testList.append(np.array(pd.read_csv(testData+file,delimiter=',',header=None)))

                scoresList.append(scorePoint(np.array(point), testList, threshold, method))
                answers.append(y+1)
                durations.append(dur)
        else:
            oldCCDir = ''
            for point, y, dur, path in dataTuples:
                ccDir = path.split('CCMatrices')[0]
                if ccDir != oldCCDir:
                    testData = ccDir+'CCMatrices/testSources/'
                    testList = []
                    for i in range(5):
                        for file in os.listdir(testData):
                            if file.rfind('S'+str(i+1)) != -1:
                                testList.append(np.array(pd.read_csv(testData+file,delimiter=',',header=None)))

                scoresList.append(np.array(point))
                answers.append(y+1)
                durations.append(dur)

        return data, scoresList, answers, durations
