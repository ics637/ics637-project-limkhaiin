README.md

# Fine-tuning pre-trained audio models for detecting dementia

This repository contains the comprehensive details of the ICS 637 final project. It includes the necessary datasets for training, validation, and testing purposes, as well as a Python script for implementation. Additionally, an attached report provides a thorough account of the project, covering aspects ranging from the training and validation process to the final evaluation.

## Dataset
The training and validation data utilized in this project were sourced from the English/Pitt dataset within the Dementia Databank (https://dementia.talkbank.org).

For the testing phase, the data was obtained from the ADRess 2021 Challenge, which is also hosted on the same website.

## Model
In this project, two pretrained models were utilized and subsequently fine-tuned.

The first model, Wav2vec, is a self-supervised model designed for Automatic Speech Recognition. Its main function is to learn the vector representation of audio data, thereby enabling effective speech analysis and processing.

The second model, Yamnet, was initially trained on a large dataset of YouTube audiosets. Its training objective involved predicting 512 event classes, which represent different types of audio. Yamnet employs a depthwise-separable convolution architecture, which contributes to its robust performance in audio classification tasks.

## Training set
The English/Pitt folders consist of two groups: control and dementia. Within each group, there are recordings of spontaneous speech during four types of tasks: cookie, fluency, recall, and sentence. The participants include individuals who have been diagnosed with Alzheimer's Disease, those who have the potential to develop it, and individuals with other forms of dementia. To create a training set, I combined all four tasks and performed an 8:2 split to generate separate training and validation sets.

## Test set

The Challenge focuses on addressing a challenging automatic prediction problem that holds societal and medical significance, specifically the detection of Alzheimer's Dementia (AD). The objective of this task is to diagnose individuals based on their spontaneous speech. To facilitate this, two distinct groups of audio sets are provided: control and dementia.

## Results

The result shows that the Wav2vec model achieved 70% auroc scores and the Yamnet model achieved 50% accuracy on a test set.

