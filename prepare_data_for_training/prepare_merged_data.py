import pandas as pd
import os


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type= str, required=True, help = 'Dataset to prepare')
parser.add_argument('--annotation', type= str, required=True, help = '')

args = parser.parse_args()


dataset = args.dataset
annotation = args.annotation

if annotation == 'tagall':
    from functions import prepare_data_iden_soft_otg_hard_tagall as prepare
elif annotation == 'idenall':
    from functions import prepare_data_iden_soft_otg_hard_idenall as prepare
elif annotation == 'onlyhate':
    from functions import prepare_data_iden_soft_otg_hard as prepare
else:
    print('You did not enter a valid annotation scheme')


out_file = 'Tagged_data_' + dataset + '_' + annotation
out_file_train = '../tagged_data/' + out_file+"/train_tagged.txt"
out_file_test = '../tagged_data/' + out_file+"/test_tagged.txt"
out_file_dev = '../tagged_data/' + out_file+"/dev_tagged.txt"

inp_data_train = "../datasets/" + dataset + "/train.tsv"
inp_data_test = "../datasets/" + dataset + "/test.tsv"
inp_data_dev = "../datasets/" + dataset + "/dev.tsv"
label_file_train = "../datasets/" + dataset + "/train_labels.tsv"
label_file_test = "../datasets/" + dataset + "/test_labels.tsv"
label_file_dev = "../datasets/" + dataset + "/dev_labels.tsv"



file_hb = "./eng_lexicon.tsv"
file_tar = './identity_combined.txt'

if not os.path.isdir('../tagged_data'):
    os.mkdir('../tagged_data')

if not os.path.isdir('../tagged_data/' + out_file):
    os.mkdir('../tagged_data/' + out_file)


print('Preparing training data...')
prepare(inp_data_train, label_file_train, file_hb, file_tar, out_file_train)
print('Preparing test data...')
prepare(inp_data_test, label_file_test, file_hb, file_tar, out_file_test)
print('Preparing dev data...')
prepare(inp_data_dev, label_file_dev, file_hb, file_tar, out_file_dev)

train_labels = pd.read_csv(label_file_train, sep = '\t', header=None, lineterminator='\n')
dev_labels = pd.read_csv(label_file_dev, sep = '\t', header=None, lineterminator='\n')
test_labels = pd.read_csv(label_file_test, sep = '\t', header=None, lineterminator='\n')

train_labels[0] = sorted(list(train_labels[0]))
dev_labels[0] = sorted(list(dev_labels[0]))
test_labels[0] = sorted(list(test_labels[0]))

train_labels.to_csv('../tagged_data/' + out_file+'/train_label_ordered.tsv', sep = '\t', header=None, index = False)
dev_labels.to_csv('../tagged_data/' + out_file+'/dev_label_ordered.tsv', sep = '\t', header=None, index = False)
test_labels.to_csv('../tagged_data/' + out_file+'/test_label_ordered.tsv', sep = '\t', header=None, index = False)

