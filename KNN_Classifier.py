### To Do ###
# Fix the fnmatch. The filename has to have the class name in them. Case sensitive!! 
# save the knn model
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018
# use the onset detection to block the audio
# try to determine the audio file

import numpy as np
import os, fnmatch
import librosa.display, librosa
from scipy.io import wavfile as read
import pickle

# Machine Learning
from sklearn import preprocessing
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score
# from sklearn.metrics import confusion_matrix, f1_score, classification_report

def getFile(path):
    files = []
    for root, dirname, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".wav"):
                files.append(os.path.join(root, file))
    print("found %d audio files"%(len(files)))
    return files

### Create label for each audio file ###
def getLabel(files):
    labels = []
    colorLabel = []
    classes = ["Crash","Kick","Tom","HiHat","Snare","Ride"] # KD,SD,HH,TT,CY,OT
    colorDict = {'Crash':'red','Kick':'orange','Tom':'yellow','HiHat':'green','Snare':'blue','Ride':'purple'}
    for fileName in files:
        for instrument in classes:
            # print(instrument)
            if fnmatch.fnmatchcase(fileName, '*'+instrument+'*'):
                # print("found file name")
                labels.append(instrument)
                colorLabel.append(colorDict[instrument])
    
    # Encode Labels for Classifier (str to number)
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(labels)

    classList = list(labelencoder.classes_)
    classNum = labelencoder.transform(labels) # encoded label (number)
    Num2class = list(labelencoder.inverse_transform(classNum)) # convert encode label back to str
    # print(len(labelencoder.classes_), "classes:", ", ".join(classList))
    # print(Num2class)
    # print(classNum)
    # print(labelencoder.classes_)
    return classNum, Num2class

### MFCC ###
def get_MFCC(y, sr=44100, blockSize=2048, hopSize=512, numMel=128, numMFCC=13):
    # Use a pre-computed log-power Mel spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=blockSize, hop_length=hopSize, n_mels=numMel)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=numMFCC)

    # Take the mean of each row
    feature_vec = np.mean(mfcc,1)
    return feature_vec

### Read File And Calculate MFCC Feature ###
def extractMFCC(files, numMFCC=13, fs=44100):
    feature_vec = np.zeros([len(files),numMFCC])
    # feature_vec_test = []
    for ind, file in enumerate(files):
        audio, sr = librosa.load(file, sr=fs)
        audio = audio/audio.max() # normalize audio file
        MFCC_feature = get_MFCC(audio)

        feature_vec[ind] = MFCC_feature
        # feature_vec_test.append(MFCC_feature) # Test

    # # Standardization to Zero-mean & Unit Variance
    # scaler = preprocessing.StandardScaler()
    # feature_vec_scaled = scaler.fit_transform(feature_vec)

    return feature_vec

### Train and Test Dataset ###
def getDatasets(path4Training, path4Testing):
    files4Training = getFile(path4Training)
    files4Testing = getFile(path4Testing)
    Training = extractMFCC(files4Training)
    Testing = extractMFCC(files4Testing)
    Training_class, Num2class_Train = getLabel(files4Training)
    Testing_class, Num2class_Test = getLabel(files4Testing)

    # print("Training Class: \n", Training_class)
    # print(files4Testing)
    # print("Testing Class: \n", Testing_class)
    # print("1st file for Training: ", files4Training[0])
    # print(Num2class_Train)
    return Training, Testing, Training_class, Testing_class

### KNN Classifier ###
def KNN(Training, Testing, Training_class):
    nearest_neighbors = 1
    KNN_model = KNeighborsClassifier(n_neighbors=nearest_neighbors)
    KNN_model.fit(Training, Training_class)
    predicted_labels = KNN_model.predict(Testing)

    ### Save the Model With Pickle
    model_filename = 'knn_model.sav'
    pickle.dump(KNN_model, open(model_filename, 'wb'))
    return predicted_labels


### Evaluation ###
def evaluation(path4Training, path4Testing, Testing_class, predicted_labels):
    # # Recall - the ability of the classifier to find all the positive samples
    # print("Recall: ", recall_score(Testing_class, predicted_labels,average=None))

    # # Precision - The precision is intuitively the ability of the classifier not to 
    # #label as positive a sample that is negative
    # print("Precision: ", precision_score(Testing_class, predicted_labels,average=None))

    # # F1-Score - The F1 score can be interpreted as a weighted average of the precision 
    # #and recall
    # print("F1-Score: ", f1_score(Testing_class, predicted_labels, average=None))

    # Accuracy - the number of correctly classified samples
    print("Accuracy: %.2f  ," % accuracy_score(Testing_class, predicted_labels,normalize=True), accuracy_score(Testing_class, predicted_labels,normalize=False) )
    print("Number of samples:",Testing_class.shape[0])


path4Training = "ACA Final Project/Training"
path4Testing = "ACA Final Project/Testing"
Training, Testing, Training_class, Testing_class = getDatasets(path4Training, path4Testing)
predicted_labels = KNN(Training, Testing, Training_class)
print('predicted:', predicted_labels)

evaluation(path4Training, path4Testing, Testing_class, predicted_labels)




# files = getFile("ACA Final Project/Test2")
# path4Training = "ACA Final Project/Training"
# path4Testing = "ACA Final Project/Test2"
# oneshot_MFCC = extractMFCC(files, numMFCC=13, fs=44100)
# # print(oneshot_MFCC)
# Training, Testing, Training_class, Testing_class = getDatasets(path4Training, path4Testing)
# predicted_labels = KNN(Training, oneshot_MFCC, Training_class)
# print(predicted_labels)


### To Do ###
# Elimate minor drum instruments, keep only KD, SN, 
# find the corresponding label for the KNN classifer
# save the classifer
# Test out the classifer using the onset time outputed
# Create more classifers (3 for each instruments)