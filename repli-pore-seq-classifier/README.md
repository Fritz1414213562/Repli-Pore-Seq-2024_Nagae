# Repli-pore-seq classifier
This software is designed to judge incoporation of nucleotide analogue (e.g. biotin-dUTP) in single-stranded DNA measured by the Oxford Nanopore Technologies devices (e.g. MinION, PromethION). The input electric current signals are fit based on their stepwise change and converted into arrays of fitting paramters (current mean, standard deviation, and duration). The arrays are converted into a floating value within 0 to 1 by the neural network. The network consists of residual convolutional neural network and LSTM network.

## prepare dataset
```
python3 path-to-src/main.py preparation --data_dir [FAST5_DIR]... (or [POD5_DIR]...) --label0 ALIGNMENT_LABEL0 --label1 ALIGNMENT_LABEL1 --output OUTPUT
```

## training NN
```
python3 path-to-src/main.py train --train_set TRAINING_SET --output MODEL_NAME
```

## fragmentation of current signals
```
python3 path-to-src/main.py fragment --data_dir [FAST5_DIR]... (or [POD5_DIR]...) --alignment ALIGNMENT --save_dir SAVE_DIR
```

## classification of fragment arrays (events)
```
python3 path-to-src/main.py classify --event_dir EVENT_DIR --model_dir MODEL_NAME --alignment ALIGNMENT --output OUTPUT
```
