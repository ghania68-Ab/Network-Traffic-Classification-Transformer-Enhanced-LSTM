# Network Traffic Classification Using Transformer-Enhanced LSTM Models

## Project Description

This project implements a Transformer-Enhanced LSTM model for network traffic classification. The model is designed to classify different types of network traffic by learning both sequence-based patterns and important relationships between traffic features.

The project uses two public datasets: CIC-Darknet2020 and UNSW-NB15. CIC-Darknet2020 focuses on darknet, Tor, VPN, and non-VPN traffic, while UNSW-NB15 focuses on normal and attack-based network traffic. Using both datasets helps evaluate the model on different network traffic scenarios.

Published baseline results from the related research paper are used only for discussion and comparison. Baseline models are not trained inside this project.

## Dataset Links

- CIC-Darknet2020: [https://www.kaggle.com/datasets/dhoogla/cicdarknet2020](https://www.kaggle.com/datasets/dhoogla/cicdarknet2020)
- UNSW-NB15: [https://www.kaggle.com/datasets/dhoogla/unswnb15](https://www.kaggle.com/datasets/dhoogla/unswnb15)

## Installation Steps

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How To Run The Code

Run CIC-Darknet2020:

```bash
python src/train.py --dataset cic --epochs 50 --batch-size 128
```

Run UNSW-NB15:

```bash
python src/train.py --dataset unsw --epochs 50 --batch-size 128
```

Run both datasets:

```bash
python src/train.py --all-datasets --epochs 50 --batch-size 128
```

Run from notebook:

```bash
jupyter notebook notebooks/experiment.ipynb
```
OR

by uploading it on Googe Colab Notebook.

## Model Details

The proposed Transformer-Enhanced LSTM uses feature preprocessing, learned feature projection, bidirectional LSTM, multi-head self-attention, residual connections, layer normalization, dense layers, dropout, regularization, AdamW optimization, and a softmax output layer.

The LSTM part helps the model learn sequential traffic patterns, while the Transformer attention part helps the model focus on important relationships between traffic features. This makes the model suitable for complex network traffic classification tasks.

## Results Summary

The model is evaluated using accuracy, precision, recall, F1-score, AUC, classification report, and confusion matrix. The results show that the proposed Transformer-Enhanced LSTM performs well on network traffic classification and is able to separate different traffic classes effectively.

Metrics are saved in the `results/` folder, while accuracy and loss curves are saved in the `figures/` folder. These generated outputs should be used as the final experimental results.

## Team Members

- Ghania Jawed (62745)
- Samia Shahzad (64248)
- Laraib Ali (65132)