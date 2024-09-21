import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import keras
from keras import Sequential
from keras import layers
from os import path
from keras.saving import load_model
import argparse
from keras.utils import to_categorical
from keras import optimizers
from sklearn.utils import shuffle
from collections import Counter

def get_patients(plist):
    def get_patient(patientid):
        rec = np.load(path.join("numpy", f"patient_{patientid+1}.npy"))
        ann = np.load(path.join("numpy", f"annotation_{patientid+1}.npy"))
        
        return rec, ann
    X, y = get_patient(plist[0])
    siglen = len(y)
    plist = plist[1::]
    for i in plist:
        rec, ann = get_patient(i)
        X = np.hstack((X, rec))
        y = np.hstack((y, ann))
        siglen += len(ann)
    
    X = np.array(np.split(X, siglen, axis=1))
    X = np.array([rec.T for rec in X])
    return X, y

def shuffle_group(X, y, unit, seed=22022009):
    np.random.seed(seed)
    size = len(X)
    resX = []
    resy = []
    idx = np.arange(size//unit)
    np.random.shuffle(idx)
    for i in idx:
        for j in range(unit):
            resX.append(X[i*unit+j])
            resy.append(y[i*unit+j])
    
    return np.array(resX), np.array(resy)

def smoothing(y, units):
    size = len(y)
    ans = np.array(np.split(np.array(y), size // units))
    ans = np.mean(ans, axis=1)
    ans = ans.flatten()
    return np.array(ans)

records = ["a01r", "a02r", "a03r", "a04r", "b01r", "c01r", "c02r", "c03r"]
batch_size = 256
epochs = 4
save_path = path.join("res", "model.keras")

parser = argparse.ArgumentParser(description='Train and evaluate on multiple patients')
parser.add_argument("-f", "--fit", help="Fit data (patient id)", required=True)
parser.add_argument("-e", "--eval", help="Evaluate data (patient id)", required=True)
args = parser.parse_args()
patient_list = args.fit.split(",")
if patient_list[-1] == "":
    patient_list.pop()
patient_list = [int(x)-1 for x in patient_list]
eval_list = args.eval.split(",")
if eval_list[-1] == "":
    eval_list.pop()
eval_list = [int(x)-1 for x in eval_list]


if patient_list != []:
    model = Sequential([
        layers.Input(shape=(1000, 2)),
        layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        # using data from last 300 seconds
        layers.SimpleRNN(30, return_sequences=True),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer="adam", 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    X, y = get_patients(patient_list)
    X, y = shuffle_group(X, y, 6)
    count = Counter(y)
    print("Apnea cases [1]:", count[1], "Normal cases [0]:", count[0])
    model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=False,
    )
    model.save(save_path)

if eval_list != []:
    model = load_model(save_path)
    X, y = get_patients(eval_list)
    X, y = shuffle(X, y, random_state=22022009)
    model.evaluate(X, y, batch_size=batch_size)