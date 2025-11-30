Remote Sensing Image Captioning (CNN + LSTM + Attention + GloVe)

This repository contains the implementation of a Remote Sensing Image Captioning model using a ResNet-50 CNN encoder, a BiLSTM decoder, Bahdanau additive attention, and GloVe 6B-300D word embeddings. The complete training and evaluation workflow is provided in a single Jupyter Notebook.

Overview

The goal of this project is to generate natural language captions for remote sensing images. Unlike natural images, remote sensing data contains complex spatial structures and multi-scale patterns. To handle this, the model combines visual feature extraction, attention-based alignment, and recurrent sequence generation. The notebook trains and evaluates the model on the RSICD dataset.

Architecture

Encoder: ResNet-50 pretrained on ImageNet-1K is used for extracting 7×7×2048 spatial feature maps. The final classification layers are removed.

Attention: Bahdanau additive attention computes alignment scores between decoder hidden states and spatial encoder features using:

e_i = vᵀ tanh(W_e * encoder_i + W_d * decoder_t)
α_i = softmax(e_i)
context_t = Σ α_i * encoder_i

Decoder: A BiLSTM decoder generates captions word-by-word. Pretrained GloVe 6B-300D embeddings are used for better semantic representation.

Repository Structure
RemoteSensing_Captioning.ipynb   # Full model training and inference pipeline
README.md

Dataset

The RSICD dataset is used for training and evaluation. It contains around 10,000 remote sensing images, each with five human-annotated captions. The images cover a wide variety of land-use categories such as airports, industrial areas, ports, rivers, and residential regions.

Evaluation Metrics

The notebook computes standard captioning metrics including BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L, and CIDEr.
Typical results achieved include:

BLEU-4 ≈ 0.40
CIDEr ≈ 2.1 – 2.7
METEOR ≈ 0.56 – 0.57

Requirements

Python 3.x and the following packages are required:

torch
torchvision
numpy
pandas
nltk
tqdm
pillow
matplotlib

GloVe embeddings (glove.6B.300d.txt) must be downloaded and placed in the specified directory inside the notebook.

Usage

Clone the repository:

git clone https://github.com/your-username/your-repo.git
cd your-repo


Open the Jupyter Notebook:

RemoteSensing_Captioning.ipynb


Set dataset and embedding paths, run the training cells, and use the inference section to generate captions for any remote sensing image.

References

Vinyals et al., "Show and Tell: A Neural Image Caption Generator"
Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
Pennington et al., "GloVe: Global Vectors for Word Representation"
RSICD Dataset

If you want a shorter version, a longer research-style version, or one tailored exactly to your repo name, tell me.
