# Speaker Diarization using Virtual Microphones

This repository contains the code to improve the classification subsystem in the work of Antonio Gomez in *Speaker Diarization and Identification from Single-Channel Classroom Audio Recording using Virtual Microphones,* the original system this work is based on. This approach is discussed in *Speaker Diarization of Noisy Classrooms from a Single Microphone Based on an Array of Virtual Microphones and Machine Learning*.

The data.zip file contains all of the data used in the project. Permission will have to be given to acces them, as they are password protected for the anonymity of the speakers. 

dataSets.py contains any code used to process the data, including the PyTorch dataset class, the train/test/val subset splitting function, and the data point scoring functions.

augmentAudio.py creates an augmented data set using PyTorch's sox interface. 

speakerClassifier.py, speakerTransformerClassifier.py and linearClassifier.py contain PyTorch neural networks, using "speaker learner" modules, transformers, and only linear layers, respectively. None of them work very well.

trainAndTest.py contain the train and test loops for the networks, which are used in train.py. All versions of the network can be trained in it, which one is selected by the netType variable.

models/ contains the best results for each tested system using each method of data input and each of the three testing methods discussed in the paper. 
NOTE: If attempting to use the SciKit-Learn models, make sure the correct version of the library is being used - i.e., the one listed in condaEnvironment.yml - as models from different version of SciKit-Learn are completely incompatible. 

scoreAndMatrix.py tests other machine learning schemes on the data, including SVMs, KNNs, random forests and XGBoost. It can also test the accuracy of the original method on all of the data.

voting.py contains the code used to develop the voting classifier. 

scratch.py contains misc python work such as graphing results. 

condaEnvironment.yml is a copy of the conda environment, for convenience. 
