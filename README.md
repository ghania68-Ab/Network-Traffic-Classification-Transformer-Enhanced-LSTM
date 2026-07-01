# Network Traffic Classification Using Transformer-Enhanced LSTM Models

## Project Description

This project implements and evaluates one proposed deep learning model: a Transformer-Enhanced LSTM for network traffic classification. The model is trained on CIC-Darknet2020 and UNSW-NB15 using a leakage-safe preprocessing pipeline, validation-based training control, class-imbalance handling, and reproducible TensorFlow/Keras training.

Published baseline results from the related research paper are used only for discussion and comparison in the paper. Baseline models are not trained inside this project.

## Dataset Links

- CIC-Darknet2020: [https://www.kaggle.com/datasets/dhoogla/cicdarknet2020](https://www.kaggle.com/datasets/dhoogla/cicdarknet2020)
- UNSW-NB15: [https://www.kaggle.com/datasets/dhoogla/unswnb15](https://www.kaggle.com/datasets/dhoogla/unswnb15)

## Installation Steps

```bash
pip install -r requirements.txt
```

## How To Run The Code

Run CIC-Darknet2020:

```bash
python src/train.py --dataset cic
```

Run UNSW-NB15:

```bash
python src/train.py --dataset unsw
```

Run both datasets:

```bash
python src/train.py --all-datasets
```

Optional overrides:

```bash
python src/train.py --dataset cic --epochs 100 --batch-size 256
```

Run from notebook:

```bash
jupyter notebook notebooks/experiment.ipynb
```

## Model Details

The proposed Transformer-Enhanced LSTM uses grouped multivariate feature timesteps, multi-scale convolutional mixing, stacked bidirectional LSTM encoders, a three-block Transformer encoder, squeeze-and-excitation channel attention, dual pooling, mixup augmentation, RobustScaler preprocessing, minority oversampling, correlation/variance feature filtering, AdamW optimization, cosine warmup scheduling, and dataset-aware class weighting.

## Results Summary

The model is evaluated using accuracy, weighted precision, weighted recall, weighted F1-score, macro precision, macro recall, macro F1-score, AUC where applicable, classification report, and confusion matrix. Metrics are saved in the `results/` folder, while accuracy and loss curves are saved in the `figures/` folder.

The generated results should be used as the final experimental results. No metrics are hard-coded or manually modified.

## Team Members

- Ghania Jawed (62745)
- Samia Shahzad (64248)
- Laraib Ali (65132)