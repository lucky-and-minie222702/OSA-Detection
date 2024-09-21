# OSA detection

## Using the CLI

```zsh
python3 src/model.py -h
```

## Data

### Preprocess data

```zsh
python3 src/preprocess.py
```

### Data used:

Dataset: [Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/)

#### Patients used

1. a01r
1. a02r
1. a03r
1. a04r
1. b01r
1. c01r
1. c02r
1. c03r
    ##### Patients without OSA: 6, 7, 8

## Final decision

### Train

```zsh
python3 src/model.py --fit "1, 2, 6, 7" --eval ""
```

### Evaluation on unseen data

```
accuracy: 0.9074 - loss: 0.3199
```
