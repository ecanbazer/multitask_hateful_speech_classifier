import pandas as pd

train_data = "Semeval_train.tsv"
dev_data = "Semeval_dev.tsv"
test_data = "Semeval_test.tsv"

train_labels = "Semeval_train_labels.tsv"
dev_labels = "Semeval_dev_labels.tsv"
test_labels = "Semeval_test_labels.tsv"

with open(train_data) as f:
  train_data = list(f)

train_data = [d[:-1] for d in train_data]

with open(dev_data) as f:
  dev_data = list(f)

dev_data = [d[:-1] for d in dev_data]

with open(test_data) as f:
  test_data = list(f)

test_data = [d[:-1] for d in test_data]

with open(train_labels) as f:
  train_labels = list(f)

train_labels = [d[:-1] for d in train_labels]

with open(dev_labels) as f:
  dev_labels = list(f)

dev_labels = [d[:-1] for d in dev_labels]

with open(test_labels) as f:
  test_labels = list(f)

test_labels = [d[:-1] for d in test_labels]

train_tag = ['train']*len(train_data)
train_dict = {'text':train_data,'label':train_labels,'exp_split':train_tag}
train_df = pd.DataFrame(data=train_dict)

dev_tag = ['dev']*len(dev_data)
dev_dict = {'text':dev_data,'label':dev_labels,'exp_split':dev_tag}
dev_df = pd.DataFrame(data=dev_dict)

test_tag = ['test']*len(test_data)
test_dict = {'text':test_data,'label':test_labels,'exp_split':test_tag}
test_df = pd.DataFrame(data=test_dict)

df_concat = pd.concat([train_df,dev_df,test_df])

df_concat.to_csv('HateEval_dataset.csv',sep=',',header=True,index=False)

