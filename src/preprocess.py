import numpy as np
import wfdb
from os import path
import os
import sys

if not path.isdir("numpy"):
    os.makedirs("numpy")

records = ["a01r", "a02r", "a03r", "a04r", "b01r", "c01r", "c02r", "c03r"]

unit = int(sys.argv[1])

for i in range(len(records)):
    rec = wfdb.rdsamp(path.join("database", records[i]))
    ann = wfdb.rdann(path.join("database", records[i]), extension="apn").symbol
    ann = np.array([[1]*unit if x == "A" else [0]*unit for x in ann])
    ann = ann.flatten()
    ann.resize(len(ann)-unit)
    info = rec[1]
    siglen = len(ann) * (6000//unit)
    rec = rec[0][:siglen:].T
    
    # remove outlier
    buffer = 30 # minutes
    rec = rec.T
    rec = rec[buffer*6000::]
    rec = rec[:len(rec)-buffer*6000:]
    ann = ann[buffer*unit::]
    ann = ann[:len(ann)-buffer*unit:]
    rec = rec.T[2::]
    # scale
    rec[1] /= 100.0

    np.save(path.join("numpy", f"patient_{i+1}"), rec)
    np.save(path.join("numpy", f"annotation_{i+1}"), ann)
    print(f"Converting patient {i+1} done!")