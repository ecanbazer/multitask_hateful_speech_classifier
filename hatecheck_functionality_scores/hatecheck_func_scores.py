from sklearn.metrics import accuracy_score as acc
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--data', type= str, required=True, help = 'Dataset')
args = parser.parse_args()

approaches = ['M0.2', 'M1.2', 'M1.2.1','M1.2.4', 'M1.2.9', 'M1.2.19', 'M2.2', 'M2.2.1', 'M2.2.4','M2.2.9','M2.2.19', 'M3.2', 'M3.2.1', 'M3.2.4','M3.2.9','M3.2.19']
funcs = ['slur_homonym_nh','slur_reclaimed_nh', 'profanity_nh','ident_neutral_nh','ident_pos_nh']
dataset= args.data

df_hatecheck = pd.read_csv('hatecheck_test_suite_cases.csv')
functions = list(df_hatecheck['functionality'])
gold = [1 if i=='hate' else 0 for i in list(df_hatecheck['label_gold'])]
sents = list(df_hatecheck['test_case'])

df_res = pd.DataFrame()
df_res['functionality'] = funcs

for approach in approaches:
    df= pd.read_csv('../inferences/Hatecheck_preds_' + approach + '_' +dataset +'.tsv', sep='\t')
    df['func'] = functions
    df['gold'] = gold
    acc42 = []
    acc15 = []
    acc31 = []
    acc67 = []
    acc83 = []
    acc_avg = []
    for i in range(len(funcs)):
        acc42.append(acc(df[df['func'] == funcs[i]]['gold'], df[df['func'] == funcs[i]]['seed42_class']))
        acc15.append(acc(df[df['func'] == funcs[i]]['gold'], df[df['func'] == funcs[i]]['seed15_class']))
        acc31.append(acc(df[df['func'] == funcs[i]]['gold'], df[df['func'] == funcs[i]]['seed31_class']))
        acc67.append(acc(df[df['func'] == funcs[i]]['gold'], df[df['func'] == funcs[i]]['seed67_class']))
        acc83.append(acc(df[df['func'] == funcs[i]]['gold'], df[df['func'] == funcs[i]]['seed83_class']))
        acc_avg.append(round(np.mean([acc42[i], acc15[i],acc67[i], acc83[i],acc31[i]])*100,1))
    df_res[approach] = acc_avg


print(df_res)

