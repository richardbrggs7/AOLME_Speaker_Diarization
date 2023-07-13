#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:38:52 2023
augmentAudio.py
@author: richard
"""

"""
Effects:
    dither ?
    echo
    reverse ?
    speed ?
    tempo
    F.add_noise ?
"""

import glob
import os
import random
import shutil
import torch
import torchaudio
import torchaudio.functional as F

# If remake is true, will delete any augmented data sets if they already exist
# and remake them, otherwise it will skip them
remake = True
dataPath = '/home/richard/documents/school/2022Fall/data'
datasets = ['LIZ_E', 'Ivone_E_geo2', 'LIZ_E0205', 'LIZ_E0305',
            'Luis_Liz_Clipped',
            'Isaac_C2L1W_q2_02-04_part1', 'Isaac_C2L1W_q2_02-04_part2',
            'G-C1L1P-Mar02-E-Irma_q2_03-08', 'G-C1L1P-Mar02-E-Irma_q2_05-08'
            ]
augmented = [dataPath + '/' + d + '_AUG/' for d in datasets]
datasetPaths = [dataPath + '/' + d + '/' for d in datasets]

for data, aug in zip(datasetPaths, augmented):
    paths = glob.glob(data+'*.wav')
    if os.path.exists(aug) and remake == False:
        break
    elif not os.path.exists(aug):
        os.mkdir(aug)
    else:
        shutil.rmtree(aug)
        os.mkdir(aug)

    for file in paths:
        # print(file)
        audio, fs = torchaudio.load(file)

        # delay = random.randint(30, 100)
        # decay = random.uniform(.2, .6)
        # echo = [['echo', '1', '.7', str(delay), str(decay)]]
        # augment, freq = torchaudio.sox_effects.apply_effects_tensor(audio,fs,echo)

        # factor = random.uniform(.7, .85)
        # tempo = [['tempo', str(factor)]]
        # augment, freq = torchaudio.sox_effects.apply_effects_tensor(audio,fs,tempo)

        # AUG0
        # delay = random.randint(30, 100)
        # decay = random.uniform(.2, .6)
        # factor = random.uniform(.7, .85)

        # AUG1
        # delay = random.randint(30, 70)
        # decay = random.uniform(.2, .5)
        # factor = random.uniform(.8, .9)

        # AUG2
        delay = random.randint(20, 40)
        decay = random.uniform(.1, .3)
        factor = random.uniform(.85, .95)

        effect = [['tempo', str(factor)],
                  ['echo', '1', '.7', str(delay), str(decay)],
                  ['dither']]
        augment, freq = torchaudio.sox_effects.apply_effects_tensor(audio,fs,effect)

        torchaudio.save(aug+'/'+file.split('/')[-1],augment,freq)

        # break

    # break

