
from bert_sequence_tagger import SequenceTaggerBert, BertForTokenClassificationCustom
from pytorch_transformers import BertTokenizer

from bert_sequence_tagger.bert_utils import get_model_parameters, prepare_flair_corpus
from bert_sequence_tagger.bert_utils import make_bert_tag_dict_from_flair_corpus 
from bert_sequence_tagger.model_trainer_bert import ModelTrainerBert
from bert_sequence_tagger.metrics import f1_entity_level, f1_token_level
from sklearn.metrics import f1_score, accuracy_score
from pytorch_transformers import AdamW, WarmupLinearSchedule
import pandas as pd
from bert_sequence_tagger.confidence_interval import confidence_interval
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('sequence_tagger_bert')
import random
import numpy as np
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type= int, required=False, default = 10, help = 'The number of epochs')
parser.add_argument('--batch', type= int, required=False, default = 32, help = 'Batch size')
parser.add_argument('--datafolder', type= str, required=True, help = 'Relative path to the data folder')
parser.add_argument('--task1coef', type= float, required=False, default = 1, help = 'Coefficient of task 1')
parser.add_argument('--task2coef', type= float, required=False, default = 1, help = 'Coefficient of task 2')
parser.add_argument('--modelpath', type= str, required=True, default = 1, help = 'Path to save the model')
args = parser.parse_args()


batch_size = args.batch
n_epochs = args.epochs
datafolder = args.datafolder
task1coef = args.task1coef
task2coef = args.task2coef
modelpath = args.modelpath

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("cuda count: ",torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        print("cuda count: ",torch.cuda.device_count())
        torch.cuda.manual_seed_all(seed)




seeds = [42, 15, 31, 67, 83]         

f1s_task1_indata = []
f1s_task2_indata = []


acc_task2_indata = []


for seed in seeds:
        set_seed(seed=seed)

        # Loading corpus ############################

        from flair.datasets import ColumnCorpus

        data_folder = './' + datafolder
        corpus = ColumnCorpus(data_folder,
                        {0 : 'text', 1 : 'ner'},
                        train_file='train_tagged.txt',
                        test_file='test_tagged.txt',
                        dev_file='dev_tagged.txt')

        class_labels_train = list(pd.read_csv(datafolder + '/train_label_ordered.tsv', sep = '\t', header=None, lineterminator='\n')[0])
        class_labels_val = list(pd.read_csv(datafolder + '/dev_label_ordered.tsv', sep = '\t', header=None,lineterminator='\n')[0])
        class_labels_test = list(pd.read_csv(datafolder + '/test_label_ordered.tsv', sep = '\t', header=None, lineterminator='\n')[0])
        # Creating model ############################


        model_type = 'bert-base-uncased'
        bpe_tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=True)

        idx2tag, tag2idx = make_bert_tag_dict_from_flair_corpus(corpus)

        model = BertForTokenClassificationCustom.from_pretrained(model_type, 
                                                                num_labels=len(tag2idx)).cuda()

        seq_tagger = SequenceTaggerBert(bert_model=model, bpe_tokenizer=bpe_tokenizer, 
                                        idx2tag=idx2tag, tag2idx=tag2idx,class_labels = class_labels_train, max_len=128,
                                        pred_batch_size=batch_size)


        # Training ############################

        train_dataset = prepare_flair_corpus(corpus.train)
        val_dataset = prepare_flair_corpus(corpus.dev)


        train_dataset_new = []
        for i in range(len(train_dataset)):
                z = train_dataset[i] + (class_labels_train[i],)
                train_dataset_new.append(z)


        val_dataset_new = []
        for i in range(len(val_dataset)):
                z = val_dataset[i] + (class_labels_val[i],)
                val_dataset_new.append(z)

        test_dataset = prepare_flair_corpus(corpus.test)

        test_dataset_new = []
        for i in range(len(test_dataset)):
                z = test_dataset[i] + (class_labels_test[i],)
                test_dataset_new.append(z)



#        optimizer = AdamW(get_model_parameters(model), lr=5e-5, betas=(0.9, 0.999), 
#                        eps=1e-6, weight_decay=0.01, correct_bias=True)

        optimizer = AdamW(get_model_parameters(model), lr=2e-5, eps=1e-8)


        n_iterations_per_epoch = len(corpus.train) / batch_size
        n_steps = n_iterations_per_epoch * n_epochs
        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0.1, t_total=n_steps)



        trainer = ModelTrainerBert(model=seq_tagger, 
                                optimizer=optimizer, 
                                lr_scheduler=lr_scheduler,
                                train_dataset=train_dataset_new, 
                                val_dataset=val_dataset_new,
                                keep_best_model = True,
                                validation_metrics=[f1_entity_level],
                                batch_size=batch_size)


        trainer.train(epochs=n_epochs, task1coef = task1coef, task2coef = task2coef)

        # Testing on in-dataset test set ############################


        class_preds, tag_preds, ___, test_metrics, _, __ = seq_tagger.predict(test_dataset_new, evaluate=True, 
                                                metrics=[f1_entity_level, f1_token_level])
        
        
        f1s_task1_indata.append(test_metrics[1])
        print(f'In-dataset F1 Task1 on seed {seed}: {test_metrics[1]}')
        print(f'In-dataset F1 Task 2 on seed {seed}: {test_metrics[3]}')
        acc = accuracy_score(class_labels_test,class_preds)
        print(f'In-dataset Accuracy Task 2 on seed {seed}: {acc}')
        acc_task2_indata.append(acc)
        f1s_task2_indata.append(test_metrics[3])
        print()
        alpha, lower, upper = confidence_interval(class_labels_test, class_preds)
        plusminus = upper - lower
        print(f' {alpha} Confidence interval on test set with seed {seed}: {lower}-{upper}, plus-minus: {plusminus}')



        #save the model
        seq_tagger.save_serialize(f'./models/{modelpath}_seed{seed}')



print('Average F1 of task 1 in-data: ', str(np.mean(f1s_task1_indata)))
print('Average F1 of task 2 in-data: ', str(np.mean(f1s_task2_indata)))
print()
print('Average Accuracy of task 2 in-data: ', str(np.mean(acc_task2_indata)))
