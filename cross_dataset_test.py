import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from redditscore import tokenizer
crazy_tokenizer = tokenizer.CrazyTokenizer(remove_punct = False, normalize=2,lowercase=True,decontract=True,urls='',hashtags='split',remove_breaks=True,latin_chars_fix=True,subreddits='')
import pandas as pd
from string import punctuation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as prec
import numpy as np

device = torch.device('cuda')
n_gpu = torch.cuda.device_count()

if not os.path.isdir('inferences'):
    os.mkdir('inferences')

if not os.path.isdir('final_scores'):
    os.mkdir('final_scores')

def remove_some_punc(s):
    puncs = punctuation.replace(',', '').replace('.','').replace("'", "").replace('"', '').replace('!','').replace('?', '')
    for char in s:
        if char in puncs:
            s = s.replace(char, '')
    return s.lower()

for i in range(n_gpu):
    print(torch.cuda.get_device_name(i))

from bert_multitask_classifier.bert_sequence_tagger.bert_for_token_classification_custom import BertForTokenClassificationCustom as BFTC
from bert_multitask_classifier.bert_sequence_tagger import SequenceTaggerBert as SeqTag

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type= str, required=True, help = 'The model')
args = parser.parse_args()
model = args.model
approaches = ['M0.2', 'M1.2', 'M1.2.1','M1.2.4', 'M1.2.9', 'M1.2.19', 'M2.2', 'M2.2.1', 'M2.2.4','M2.2.9','M2.2.19', 'M3.2', 'M3.2.1', 'M3.2.4','M3.2.9','M3.2.19']
testsets = ['Hateval', 'Davidson', 'Founta', 'Vidgen', 'Waseem', 'Hatecheck', 'HatevDavFou', 'HatevDavFouWas']

df_scores = pd.DataFrame()
df_scores['test_sets'] = testsets

for approach in approaches:
    model1 = SeqTag.load_serialized('bert_multitask_classifier/models/model_' + approach + '_' + model + '_seed42', BFTC)
    model2 = SeqTag.load_serialized('bert_multitask_classifier/models/model_' + approach + '_' + model + '_seed15', BFTC)
    model3 = SeqTag.load_serialized('bert_multitask_classifier/models/model_' + approach + '_' + model + '_seed31', BFTC)
    model4 = SeqTag.load_serialized('bert_multitask_classifier/models/model_' + approach + '_' + model + '_seed67', BFTC)
    model5 = SeqTag.load_serialized('bert_multitask_classifier/models/model_' + approach + '_' + model + '_seed83', BFTC)
    models = {model1:'seed42',model2: 'seed15', model3: 'seed31', model4: 'seed67', model5: 'seed83'}
    approach_f1s = []
    approach_accs = []
    for testset in testsets:
        sents = list(pd.read_csv('datasets/' +testset +'/test.tsv', sep = '\t', header = None)[0])
        gold = list(pd.read_csv('datasets/' +testset+ '/test_labels.tsv', sep = '\t', header = None)[0])
        sents_ = [remove_some_punc(' '.join(crazy_tokenizer.tokenize(i))) for i in sents]
        f1s = []
        accs = []
        

        df = pd.DataFrame()
        df['sents'] = sents
        for checkpoint in models.keys():
            preds, tags, __ = checkpoint.predict([sent.split(" ") for sent in sents_])
            f1s.append(f1_score(gold,preds, average = 'macro')) 
            accs.append(acc(gold,preds))
            df[models[checkpoint] + '_tags'] = tags
            df[models[checkpoint] + '_class'] = preds
            df.to_csv('inferences/' + testset + '_preds_' + approach + '_' + model + '.tsv', sep = '\t')
 #           df_scores[approach + '_acc'] = accs
#            df_scores[approach + '_f1'] = f1s
#            print(df_scores)


        f1 = round(np.mean(f1s)*100, 1)
        accur = round(np.mean(accs)*100,1)
        approach_f1s.append(f1)
        approach_accs.append(accur)
    df_scores[approach + '_acc'] = approach_accs
    df_scores[approach + '_f1'] = approach_f1s
print(df_scores)
df_scores.to_csv('scores/' + model  +'.tsv', sep  = '\t')
