
# Dynamic Attention Mechanism for Adaptive Knowledge Distillation in Speech Recognition 

#### Overview:

This project aims to improve speech recognition capabilities using a Dynamic Attention Mechanism for Adaptive Knowledge Distillation. The focus is on enhancing the adaptability and accuracy of speech recognition models in dynamic acoustic environments, critical for applications in cyber-physical systems.




## Deployment

#### Environment Setup:

Programming Language: Python 3.8+
Required Tools: Google Colab or any environment supporting PyTorch and torchaudio.


#### Data:

The model is trained and evaluated using the LibriSpeech dataset, which is a publicly available collection of approximately 1000 hours of 16kHz English speech. You can obtain the dataset by enabling automatic download in the training script or by visiting LibriSpeech Dataset at this link: https://www.openslr.org/12



To deploy this project run

```bash
  npm run deploy
```


## Dependencies

Python 3.8+

PyTorch 1.11.0

torchaudio 0.11.0

comet-ml 3.0.2

numpy

matplotlib
## Code Structure
#### 1. Utility Functions

The script includes utility functions for calculating the Levenshtein distance and error rates:

Levenshtein Distance (_levenshtein_distance): Calculates the minimum edits required to transform one string into another, suitable for comparing word sequences.

Word Errors (word_errors): Computes the edit distance at the word level between two sentences.

Character Errors (char_errors): Calculates the edit distance at the character level.

WER Calculation (wer): Utilizes the word-level edit distances to calculate the Word Error Rate (WER).

CER Calculation (cer): Uses character-level edit distances to calculate the Character Error Rate (CER).

#### 2. Text Transformation Class (TextTransform):

This class provides methods to convert text to integers and vice versa, using a predefined character map. It's essential for encoding text data into a format suitable for neural network input.

#### 3. Data Preprocessing (data_processing):

This function processes the audio data by applying transformations and converting labels with the TextTransform class. It efficiently manages both training and validation data types.

#### 4. Model Definition:

The script defines a neural network architecture with various components:

CNN Layer Normalization (CNNLayerNorm): Applies layer normalization tailored for CNN inputs.

Residual CNN (ResidualCNN): Consists of CNN layers with residual connections and normalization to enhance feature learning.

Bidirectional GRU (BidirectionalGRU): Implements a GRU layer that processes data bidirectionally, enhancing sequence learning.

Speech Recognition Model (SpeechRecognitionModel): Integrates CNNs, RNNs (GRUs), and a classifier layer, designed to extract features from audio data and predict text outputs.

#### 5. Training and Evaluation Functions:

Train Function (train): Manages the training loop, updating model weights based on the computed losses.

Test Function (test): Evaluates the model on the test dataset, computing losses and error rates.

Checkpointing Function (checkpoint): Saves the model state, facilitating the resumption of training or model reloading later.

#### 6. Main Execution Function (main):

This function sets up the model, data loaders, and the training/test loops. It initializes all parameters, prepares datasets, and initiates the training process. It also manages the plotting of training and test results.

#### 7. Execution Block:

At the end of the script, the main function is invoked with specified hyperparameters, triggering the entire workflow of the model from training to evaluation.
## Configuration Files

No external configuration files are used. All settings are passed as command-line arguments or are hardcoded in the scripts.
## Output Interpretation

Outputs include the training and test losses, as well as the Word Error Rate (WER) and Character Error Rate (CER). These metrics help evaluate the model’s transcription accuracy. Lower values indicate better performance.