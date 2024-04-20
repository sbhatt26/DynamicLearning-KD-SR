# Speech Recognition with Wav2Vec2

This is the README file for our Model Number 2.
This repository contains code for speech recognition using Wav2Vec2, a pretrained model for Automatic Speech Recognition (ASR) released by Hugging Face's Transformers library.

## Overview

The code provided here allows you to:
- Perform speech recognition on audio data using Wav2Vec2 model.
- Compute Word Error Rate (WER) and Character Error Rate (CER) metrics.
- Implement a Greedy Decoder for decoding model outputs.
- Train a student model for speech recognition using knowledge distillation from a pre-trained Wav2Vec2 teacher model.
- Evaluate the performance of the trained student model on test data.
- Perform inference on new audio samples using the trained student model

## Requirements

To run the code, you need the following dependencies:
- Python 3.x
- PyTorch
- torchaudio
- transformers
- torchaudio
- soundfile
- comet_ml
- matplotlib
- librosa

## Usage
 Import the necessary libraries:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer

learning_rate = 5e-4
batch_size = 20
epochs = 10
libri_train_set = "dev-clean"
libri_test_set = "test-clean"
experiment = Experiment(
  api_key="your_comet_ml_api_key",
  project_name="your_project_name",
  workspace="your_workspace",
  log_code=True
)

main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set, experiment=experiment)

# Optionally, make translations and play audio samples
play_audio()
