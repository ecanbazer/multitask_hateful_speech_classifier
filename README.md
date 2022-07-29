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

The within-dataset scores for each task is output as well as the best performing checkpoint. Example usage, inside *bert_multitask_classifier* repo:


```python
python train.py --datafolder PATH_TO_TAGGED_DATA --modelpath PATH_TO_SAVE_MODEL 

```
## Cross-dataset evaluation

The saved model can be further evaluated on the test sets of other datasets. Example usage:

```python
python cross_dataset_test.py --model DATASET_USED_FOR_TRAINING
```
The script gets the named of the dataset used to train the model. It outputs the predictions for each test inside *inferences* repo and the Accuracy and Macro-F1 scores inside *final_results* folder. 
