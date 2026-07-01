# Network Traffic Classification Using Transformer-Enhanced LSTM Models

## Project Description

<<<<<<< HEAD
This project implements and evaluates one proposed deep learning model: a Transformer-Enhanced LSTM for network traffic classification. The model is trained on CIC-Darknet2020 and UNSW-NB15 using a leakage-safe preprocessing pipeline, validation-based training control, class-imbalance handling, and reproducible TensorFlow/Keras training.

Published baseline results from the related research paper are used only for discussion and comparison in the paper. Baseline models are not trained inside this project.
=======
This project implements a Transformer-Enhanced LSTM model for network traffic classification. The model is designed to classify different types of network traffic by learning both sequence-based patterns and important relationships between traffic features.

The project uses two public datasets: CIC-Darknet2020 and UNSW-NB15. CIC-Darknet2020 focuses on darknet, Tor, VPN, and non-VPN traffic, while UNSW-NB15 focuses on normal and attack-based network traffic. Using both datasets helps evaluate the model on different network traffic scenarios.

Published baseline results from the related research paper are used only for discussion and comparison. Baseline models are not trained inside this project.
>>>>>>> e38505b (Update project with latest changes)

## Dataset Links

- CIC-Darknet2020: [https://www.kaggle.com/datasets/dhoogla/cicdarknet2020](https://www.kaggle.com/datasets/dhoogla/cicdarknet2020)
- UNSW-NB15: [https://www.kaggle.com/datasets/dhoogla/unswnb15](https://www.kaggle.com/datasets/dhoogla/unswnb15)

## Installation Steps

<<<<<<< HEAD
=======
Install the required libraries:

>>>>>>> e38505b (Update project with latest changes)
```bash
pip install -r requirements.txt
```

## How To Run The Code

Run CIC-Darknet2020:

```bash
<<<<<<< HEAD
python src/train.py --dataset cic
=======
python src/train.py --dataset cic --epochs 50 --batch-size 128
>>>>>>> e38505b (Update project with latest changes)
```

Run UNSW-NB15:

```bash
<<<<<<< HEAD
python src/train.py --dataset unsw
=======
python src/train.py --dataset unsw --epochs 50 --batch-size 128
>>>>>>> e38505b (Update project with latest changes)
```

Run both datasets:

```bash
<<<<<<< HEAD
python src/train.py --all-datasets
```

Optional overrides:

```bash
python src/train.py --dataset cic --epochs 100 --batch-size 256
```

=======
python src/train.py --all-datasets --epochs 50 --batch-size 128
```

>>>>>>> e38505b (Update project with latest changes)
Run from notebook:

```bash
jupyter notebook notebooks/experiment.ipynb
```
OR

<<<<<<< HEAD
## Model Details

The proposed Transformer-Enhanced LSTM uses grouped multivariate feature timesteps, multi-scale convolutional mixing, stacked bidirectional LSTM encoders, a three-block Transformer encoder, squeeze-and-excitation channel attention, dual pooling, mixup augmentation, RobustScaler preprocessing, minority oversampling, correlation/variance feature filtering, AdamW optimization, cosine warmup scheduling, and dataset-aware class weighting.

## Results Summary

The model is evaluated using accuracy, weighted precision, weighted recall, weighted F1-score, macro precision, macro recall, macro F1-score, AUC where applicable, classification report, and confusion matrix. Metrics are saved in the `results/` folder, while accuracy and loss curves are saved in the `figures/` folder.

The generated results should be used as the final experimental results. No metrics are hard-coded or manually modified.
=======
by uploading it on Googe Colab Notebook.

## Model Details

The proposed Transformer-Enhanced LSTM uses feature preprocessing, learned feature projection, bidirectional LSTM, multi-head self-attention, residual connections, layer normalization, dense layers, dropout, regularization, AdamW optimization, and a softmax output layer.

The LSTM part helps the model learn sequential traffic patterns, while the Transformer attention part helps the model focus on important relationships between traffic features. This makes the model suitable for complex network traffic classification tasks.

## Results Summary

The model is evaluated using accuracy, precision, recall, F1-score, AUC, classification report, and confusion matrix. The results show that the proposed Transformer-Enhanced LSTM performs well on network traffic classification and is able to separate different traffic classes effectively.

Metrics are saved in the `results/` folder, while accuracy and loss curves are saved in the `figures/` folder. These generated outputs should be used as the final experimental results.
>>>>>>> e38505b (Update project with latest changes)

## Team Members

- Ghania Jawed (62745)
- Samia Shahzad (64248)
- Laraib Ali (65132)