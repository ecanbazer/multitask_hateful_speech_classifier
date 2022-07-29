# Multitask Hateful Speech Classifier


The multitask hateful speech classifier is a BERT based text classification model having a sequence tagging subtask for adding awareness to target group terms and profanity/slurs.


The input is in connl-like format where tokens are annotated in BIO scheme. Target group terms are tagged with "IDEN", slur terms are tagged with "OTG", other tokens are tagged as "O". Corresponding hate speech annotation (binary: hate(1)/non-hate(0)) is also provided in a separate file.

The model learns to classify hate speech, using the information acquired from the sequence tagging subtask, it outputs the predicted tags and predicted binary labels.

## Data preparation

A simple hate speech classification dataset can be transformed into the input format, automatically tagging the target identity terms and profane/slur terms in a lexicon-based way. The scrpit for preparing the data is given the path to the dataset to be prepared and the annotation scheme as command-line arguments. It outputs a folder of tagged and formatted data, ready to for training. Example usage, inside *prepare_data_for_training* repo:


```python
python prepare_data.py --dataset PATH_TO_DATASET --annotation tagall

```

## Training

Example usage, inside *
