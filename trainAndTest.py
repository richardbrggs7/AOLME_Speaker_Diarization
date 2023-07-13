#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:44:44 2022
trainAndTest.py
@author: spacer
"""

import torch

def trainLoop(net, epochs, trainData, valData, optimizer, lossFun,
              stoppingBuffer = 10, device=torch.device('cpu')):
    """
    Training loop for a PyTorch classification network

    Parameters
    ----------
    net : PyTorch network
        The network to be trained
    epochs : int
        Number of epochs
    trainData : DataLoader
        PyTorch DataLoader for training data
    valData : DataLoader
        PyTorch DataLoader for validation Data
    optimizer : PyTorch Optimizer
        Optimizer for net
    lossFun : Function Handle
        Loss function
    stoppingBuffer : int, optional
        The amount of epochs that training should wait before considering early
        stopping. The default is 10.
    cuda : Device, optional
        PyTorch device if cuda is supported for net, else cpu. The default is
        torch.device('cpu').

    Returns
    -------
    resultDict : Dict
    Dictionary containing:
        'net' : The trained network
        'trainAcc' : List history of training accuracies
        'valAcc' : List history of validation accuracies
        'trainLoss' : List history of train losses
        'valLoss' : List history of validation losses

    """

    trainLosses = []
    valLosses = []
    trainAccs = []
    valAccs = []
    lossBuffer = [100]
    for e in range(epochs):
        net.train()
        trainCorrect = 0
        valCorrect = 0
        trainTotal = 0
        valTotal = 0
        avgTrainLoss = 0
        avgValLoss = 0
        for x, y, _ in trainData:
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            optimizer.zero_grad()
            trainLoss = lossFun(pred, y)
            # trainLosses.append(trainLoss.item())
            avgTrainLoss += trainLoss.item()
            trainLoss.backward()
            optimizer.step()

            for i, _ in enumerate(y):
                trainTotal += 1
                p = torch.argmax(pred[i])
                if y[i] == p:
                    trainCorrect += 1

        trainAcc = trainCorrect/trainTotal
        trainAccs.append(trainAcc)
        avgTrainLoss /= trainTotal
        trainLosses.append(avgTrainLoss)

        net.eval()
        for x, y, _ in valData:
            x = x.to(device)
            y = y.to(device)
            pred = net(x)
            valLoss = lossFun(pred, y)
            # valLosses.append(valLoss.item())
            avgValLoss += valLoss.item()
            for i, _ in enumerate(y):
                valTotal += 1
                p = torch.argmax(pred[i])
                if y[i] == p:
                    valCorrect += 1

        valAcc = valCorrect/valTotal
        valAccs.append(valAcc)
        avgValLoss /= valTotal
        valLosses.append(avgValLoss)

        print('Epoch: {}  \tTrain Acc: {:.2f}%    Val Acc: {:.2f}%    Train Loss: {:.3f}    Val Loss: {:.3f}'
              .format(e, trainAcc*100, valAcc*100, avgTrainLoss, avgValLoss))

        if avgValLoss < min(lossBuffer):
            bestNet = net.state_dict()
            bestE = e
        if e > stoppingBuffer:
            if avgValLoss > max(lossBuffer):
                net.load_state_dict(bestNet)
                print('Best net found at epoch', bestE)
                break
            lossBuffer.pop(0)

        lossBuffer.append(avgValLoss)

    resultDict = {}
    resultDict['net'] = net.to('cpu').state_dict()
    resultDict['trainAcc'] = trainAccs
    resultDict['valAcc'] = valAccs
    resultDict['trainLoss'] = trainLosses
    resultDict['valLoss'] = valLosses

    return resultDict

def testLoop(net, testData, device=torch.device('cpu')):
    """
    Loop for testing speakerClassifier

    Parameters
    ----------
    net : PyTorch network
        The network to be tested.
    testData : PyTorch Dataloader
        Dataloader for test data.
    device : PyTorch device, optional
        Device on which the network and data are stored. The default is torch.device('cpu').

    Returns
    -------
    acc : float
        Accuracy of the network.
    errorRates : list
        List of error rates of the network relative to each class.

    """
    correct = 0
    total = 0
    numClasses = testData.dataset.classes
    actualTimes = [0] * numClasses

    predictedTimes = [0] * numClasses
    numPerClass = [0] * numClasses
    errorRates = []
    wrongTimes = 0
    for x, y, dur in testData:
        x = x.to(device)
        y = y.to(device)
        pred = net(x)
        for i, _ in enumerate(y):
            total += 1
            p = torch.argmax(pred[i])
            actualTimes[y[i].item()] += dur[i].item()
            predictedTimes[p.item()] += dur[i].item()
            numPerClass[y[i].item()] += 1
            if y[i] == p:
                correct += 1
            else:
                wrongTimes += dur[i].item()

    acc = correct/total
    der = wrongTimes/sum(actualTimes)
    print()
    print('Test Accuracy: {:.3f}%'.format(acc*100))
    print('Test DER: {:.3f}%'.format(der*100))
    print('Predicted:', predictedTimes)
    print('Actual:   ', actualTimes)
    # print('Error rates:')
    # for i in range(numClasses):
    #     error = abs(predictedTimes[i] - actualTimes[i])/actualTimes[i]
    #     print('\tSpeaker '+str(i+1)+': {:.3f}'.format(error))
    #     errorRates.append(error)
    return acc, der, errorRates
