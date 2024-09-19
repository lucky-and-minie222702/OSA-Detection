import numpy as np
import wfdb
from os import path

records = ["a01r", "a02r", "a03r", "a04r", "b01r", "c01r", "c02r", "c03r"]

for i in range(len(records)):
    rec = wfdb.rdsamp(path.join("database", records[i]))
    ann = wfdb.rdann(path.join("database", records[i]), extension="apn").symbol
    ann = np.array([1 if x == "A" else 0 for x in ann])
    ann.resize(len(ann)-1)
    info = rec[1]
    siglen = len(ann) * 6000
    rec = rec[0][:siglen:].T
    print(siglen)
    
    # remove outlier
    buffer = 30 # minutes
    rec = rec.T
    rec = rec[buffer*6000::]
    rec = rec[:len(rec)-buffer*6000:]
    ann = ann[buffer::]
    ann = ann[:len(ann)-buffer:]
    rec = rec.T[2::]
    np.save(path.join("numpy", f"patient_{i+1}"), rec)
    np.save(path.join("numpy", f"annotation_{i+1}"), ann)
    print(f"Converting patient {i+1} done!")