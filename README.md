
# Dynamic Attention Mechanism for Adaptive Knowledge Distillation in Speech Recognition 

#### Overview:

This is the README file for our Model Number 1. This project aims to improve speech recognition capabilities using a Dynamic Attention Mechanism for Adaptive Knowledge Distillation. The focus is on enhancing the adaptability and accuracy of speech recognition models in dynamic acoustic environments, critical for applications in cyber-physical systems.

Here is the GitHub repository link: https://github.com/sbhatt26/DynamicLearning-KD-SR.git




## Deployment

### Environment Setup:

Programming Language: Python 3.8+
Required Tools: Google Colab or any environment supporting PyTorch and torchaudio.


### Data:

The model is trained and evaluated using the LibriSpeech dataset, which is a publicly available collection of approximately 1000 hours of 16kHz English speech. You can obtain the dataset by running the training script or by visiting LibriSpeech Dataset at this link: https://www.openslr.org/12. Running the below code in Google Colab will download the dataset in the particular runtime.



### Running the Code in Google Colab:

#### Note:
Add all of the below code in one cell in Google Colab notebook and then run the cell.

#### Step 1: Clone the GitHub Repository

In a new cell at the top of your Google Colab notebook, enter the following command to clone the repository:

```bash
!git clone https://github.com/sbhatt26/DynamicLearning-KD-SR.git
```

#### Step 2: Change Directory
After cloning the repository, change your current working directory to the repository's folder by running:

```bash
%cd DynamicLearning-KD-SR
```
#### Step 3: Install Dependencies
Install the required Python packages specified in the requirements.txt file by running:
```bash
!pip install -r requirements.txt
```

#### Step 4: Run the Script
Execute the aiforcps_finalproject_model1.py script by running:

```bash
!python aiforcps_finalproject_model1.py
```

#### Alternative if above code doesn't work:

An alternative to above method if for some reason it doesn't work than copy the code from .ipynb file and run in your Google Colab Notebook, cell after cell as is in the notebook. The .ipynb file is named "AIFORCPS_FINALPROJECT_MODEL1.ipynb" under the GitHub repository, link given in the Overview section.
## Dependencies

Python 3.8+

PyTorch 1.11.0

torchaudio 0.11.0

comet-ml 3.0.2

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
## References

1. Chu, Shih-Chuan, Chung-Hsien Wu, and Tsai-Wei Su. "Speech Enhancement Using Dynamic Learning in Knowledge Distillation via Reinforcement Learning." IEEE Access (2023). https://ieeexplore.ieee.org/abstract/document/10363197

2. Pimentel, Arthur, et al. "Environment-Aware Knowledge Distillation for Improved Resource-Constrained Edge Speech Recognition." Applied Sciences 13.23 (2023): 12571. https://www.mdpi.com/2076-3417/13/23/12571

3. Zhang, Hailin, Defang Chen, and Can Wang. "Confidence-aware multi-teacher knowledge distillation." ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022. https://ieeexplore.ieee.org/abstract/document/9747534?casa_token=7dmvUD9mwMgAAAAA:L3O9k7wiL6m8i5TwfHR0iQqOFvx2-CoWVUPwlNl3fvIlFSrzJ8No6hWklVJc6PuXsWDt01eK

4. Amodei, Dario, et al. "Deep speech 2: End-to-end speech recognition in english and mandarin." International conference on machine learning. PMLR, 2016. https://proceedings.mlr.press/v48/amodei16.html

5. Yi, Cheng, et al. "Applying wav2vec2. 0 to speech recognition in various low-resource languages." arXiv preprint arXiv:2012.12121 (2020).

