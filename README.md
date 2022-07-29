# Multitask Hateful Speech Classifier


The multitask hateful speech classifier is a BERT based text classification model having a sequence tagging subtask for adding awareness to target group terms and profanity/slurs.


The input is in connl-like format where tokens are annotated in BIO scheme. Target group terms are tagged with "IDEN", slur terms are tagged with "OTG", other tokens are tagged as "O". Corresponding hate speech annotation (binary: hate/non-hate) is also provided in a separate file.

The model learns to classify hate speech, using the information acquired from the sequence tagging subtask, it outputs the predicted tags and predicted binary labels.

##Data preparation

