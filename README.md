# microGEXT

This model used `Python 3.8` and `Pytorch 1.7.0`.

# Training
To train a recognizer:
1. Download the microGEXT dataset.
2. Change the data paths in `train.py` to the local path on your computer.
3. Change the `model_fold` paths in `train.py` to a local directory.
4. Run `train.py`.

The best models should be saved in `model_fold`. 

# Inference
Open the `parser.py`, change `inference=False` to `inference=True`, then run `train.py`.

# Calibration
Open the `parser.py`, change `calibration=False` to `calibration=True`, then run `train.py`.

