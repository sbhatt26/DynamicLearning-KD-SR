# Speech Recognition with Wav2Vec2

This is the README file for our Model Number 2.
This repository contains code for speech recognition using Wav2Vec2, a pretrained model for Automatic Speech Recognition (ASR) released by Hugging Face's Transformers library.

## Overview

Here is the, GitHub Repository Link: https://github.com/sbhatt26/DynamicLearning-KD-SR.git
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

### Running the Code in Google Colab:

#### Note:
Add all of the below code in one cell in Google Colab notebook and then run the cell.

#### Step 1: Install the dependencies

```bash
pip install torchaudio torch comet_ml transformers librosa soundfile
```

#### Step 2: Clone the GitHub Repository

In a new cell at the top of your Google Colab notebook, enter the following command to clone the repository:

```bash
!git clone https://github.com/sbhatt26/DynamicLearning-KD-SR.git
```

#### Step 3: Change Directory
After cloning the repository, change your current working directory to the repository's folder by running:

```bash
%cd DynamicLearning-KD-SR
```

#### Step 4: Run the Script
Execute the aiforcps_finalproject_model2.py script by running:

```bash
!python aiforcps_finalproject_model2.py
```

#### Alternative if above code doesn't work:

An alternative to above method if for some reason it doesn't work than copy the code from .ipynb file and run in your Google Colab Notebook, cell after cell as is in the notebook. The .ipynb file is named "AIFORCPS_FINALPROJECT_MODEL2.ipynb" under the GitHub repository, link given in the Overview section.

## References

1. Chu, Shih-Chuan, Chung-Hsien Wu, and Tsai-Wei Su. "Speech Enhancement Using Dynamic Learning in Knowledge Distillation via Reinforcement Learning." IEEE Access (2023). https://ieeexplore.ieee.org/abstract/document/10363197

2. Pimentel, Arthur, et al. "Environment-Aware Knowledge Distillation for Improved Resource-Constrained Edge Speech Recognition." Applied Sciences 13.23 (2023): 12571. https://www.mdpi.com/2076-3417/13/23/12571

3. Zhang, Hailin, Defang Chen, and Can Wang. "Confidence-aware multi-teacher knowledge distillation." ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022. https://ieeexplore.ieee.org/abstract/document/9747534?casa_token=7dmvUD9mwMgAAAAA:L3O9k7wiL6m8i5TwfHR0iQqOFvx2-CoWVUPwlNl3fvIlFSrzJ8No6hWklVJc6PuXsWDt01eK

4. Amodei, Dario, et al. "Deep speech 2: End-to-end speech recognition in english and mandarin." International conference on machine learning. PMLR, 2016. https://proceedings.mlr.press/v48/amodei16.html

5. Yi, Cheng, et al. "Applying wav2vec2. 0 to speech recognition in various low-resource languages." arXiv preprint arXiv:2012.12121 (2020).
