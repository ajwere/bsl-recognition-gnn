British Sign Language Recognition Using Spatio-Temporal Graph Neural Networks

A computer vision and deep learning project investigating landmark-based British Sign Language (BSL) recognition using spatio-temporal graph neural networks.

Overview

This project investigates the recognition of British Sign Language signs from video using visual landmarks and graph neural networks.

MediaPipe is used to extract landmark representations of the hands, mouth and upper body. These landmarks are modelled as spatio-temporal graphs, enabling the network to learn spatial relationships between landmarks and movement patterns across video sequences.

The project focuses on landmark-based recognition, investigating graph-based learning approaches for classifying signs across a large vocabulary.

Originally developed as part of my BSc Computer Science research at Loughborough University, this work forms part of my broader research into deep learning and computer vision for sign language recognition.

Real-Time Demonstration

The video demonstrates the model performing real-time British Sign Language (BSL) recognition. During the demonstration, the model correctly predicts several signs, including Beautiful (confidence: 0.95), Flower (0.39), Change (0.88), and Job (0.46).

The confidence scores illustrate how the model’s certainty varies between predictions, highlighting both strong predictions and cases where recognition is less certain.


https://github.com/user-attachments/assets/9832de76-3333-4247-aef5-74af56700a85





Research Objectives

* Develop a landmark-based BSL recognition system using deep learning.
* Represent body landmarks as spatio-temporal graphs.
* Investigate adaptive graph learning for modelling relationships between landmarks.
* Evaluate recognition performance across a large sign vocabulary.
* Analyse classification performance using instance-level and class-level metrics.

Dataset

The project uses the BBC-Oxford BSL-1K dataset for large-scale British Sign Language recognition.

Video samples are processed into word-level clips, from which landmark features are extracted and prepared for model training and evaluation.

The dataset and associated data files are not included in this repository. Access and use remain subject to the original dataset’s licensing and access conditions.

Methodology

1. Video Preprocessing

Video data is processed into word-level clips suitable for landmark extraction and subsequent model training.

2. Landmark Extraction

MediaPipe is used to extract landmarks representing the hands, mouth and upper body. The resulting sequences provide structured spatial information across time.

3. Spatio-Temporal Graph Learning

Landmark sequences are represented as graphs. The graph neural network learns relationships between landmarks to capture spatial and temporal patterns relevant to sign recognition.

4. Model Training

The extracted landmark features are loaded and processed for training. The model is trained to classify sign sequences across the selected vocabulary.

5. Evaluation

The trained model is evaluated using classification metrics, including top-1 accuracy, top-5 accuracy and class-level performance measures.

Further analysis can include per-class results, confusion matrices and examples of commonly confused signs.

## Repository Structure

```text
bsl-recognition-gnn/
├── README.md
├── requirements.txt
├── .gitignore
│
├── train.py
│
├── assets/
│   ├── dataset_loader.py
│   ├── dataset_utils.py
│   ├── gnn_model.py
│   └── utils/
│       ├── utils.py
│       └── train_utils.py
│
├── preprocessing/
│   ├── split.py
│   ├── extract_clips.py
│   └── extract_landmarks.py
│
└── results/
    └── results.txt
    └── figures/

```

Core Files

File	Description
train.py	Main training script for the 500-class landmark-only baseline.
gnn_model.py	Defines the graph neural network architecture.
train_utils.py	Provides training-related utility functions.
dataset_loader.py	Loads landmark data for training and evaluation.
dataset_utils.py	Provides dataset preparation utilities.
utils.py	Contains general-purpose helper functions.
preprocessing/split.py	Creates training, validation and test splits.
preprocessing/extract_clips.py	Extracts word-level video clips.
preprocessing/extract_landmarks.py	Extracts landmark features from video.

The descriptions above reflect the filenames and your initial explanation. We’ll verify each file’s exact responsibilities against the code before finalising the documentation.

Technologies

* Python
* PyTorch
* PyTorch Geometric
* MediaPipe
* OpenCV
* NumPy

Experimental Results


The landmark-only baseline was evaluated on a 500-class British Sign Language recognition task using the BBC-Oxford BSL-1K dataset.

Performance

Metric	Result
Number of sign classes	500
Top-1 accuracy (per-instance)	78.40%
Top-5 accuracy (per-instance)	91.62%
Top-1 accuracy (per-class)	73.09%
Top-5 accuracy (per-class)	89.63%
Macro precision	75.33%
Macro recall	73.09%
Macro F1-score	72.70%
Evaluation loss	1.6183

Interpretation

The model achieves 78.40% top-1 accuracy, correctly predicting the target sign as its highest-ranked class for approximately 78 out of every 100 evaluated instances. Top-5 accuracy increases to 91.62%, indicating that the correct class appears among the model’s five highest-ranked predictions for approximately 92 out of every 100 instances.

The difference between instance-level and class-level performance also highlights the importance of evaluating recognition across individual sign classes, rather than relying solely on aggregate accuracy.

These results provide a baseline for further investigation into landmark-based sign recognition and spatio-temporal graph neural networks.

Results correspond to the evaluated 500-class landmark-only baseline. Evaluation settings and experimental details will be documented alongside the code.

Experimental settings, evaluation procedures and comparisons will be documented to provide context for the reported results.

Reproducibility

Reproducing the experiments requires access to the relevant dataset, compatible software dependencies and the preprocessing pipeline.

Installation instructions and commands for preprocessing, training and evaluation will be provided once the codebase has been prepared for public release.

Limitations and Future Work

Potential areas for further investigation include:

* Improving performance on difficult or visually similar signs.
* Analysing per-class errors and confusion patterns.
* Evaluating generalisation across signers and recording conditions.
* Investigating model efficiency and inference latency.
* Exploring further improvements to spatio-temporal graph learning.

Academic Context

This project was developed as part of my BSc Computer Science research at Loughborough University and contributes to my broader interest in artificial intelligence, deep learning and computer vision.
