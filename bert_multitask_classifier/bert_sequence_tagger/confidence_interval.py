from __future__ import division
import math, random
#from pycm import *
from tqdm import tqdm
import numpy
import sys
from sklearn import metrics
#file_ = 'results_predictions_augment_davidson_multiclass_augment_0.25_0.25_0.25_unique_0.tsv'
n_iterations = 10000
def sample(list_GT,list_dat):
  """Sample len(list) list elements with replacement"""

  resample_GT = []
  resample = []

  for _ in range(len(list_GT)//2):
    random_choice = random.choice(range(len(list_GT)))
    resample_GT.append(list_GT[random_choice])
    resample.append(list_dat[random_choice])

  return resample_GT, resample

def fscore(GT,data):
  """Compute F1 score"""
#  cm = ConfusionMatrix(actual_vector=GT, predict_vector=data)
#  return cm.F1_Macro
  return metrics.f1_score(GT,data,average='macro')

def confidence_interval(data_GT,data):

#  dat_GT = [line.strip().split()[0].strip() for line in open(file_)]
#  data = [line.strip().split()[1].strip() for line in open(file_)]
#
#  dat_GT = dat_GT[1:]
#  data = data[1:]
#
  print("len dat_GT:", len(data_GT))
  print("len data:", len(data))

#  print("dat_GT:", dat_GT[:5])
#  print("data:", data[:5])

  # create bootstrap distribution of F(B) - F(A)
  stats = []
  for i in range(n_iterations):
	# prepare train and test sets
    sample_GT, sample_data = sample(data_GT, data)
    stats.append(fscore(sample_GT, sample_data))
  
  # confidence intervals
  alpha = 0.95
  p = ((1.0-alpha)/2.0) * 100
  lower = max(0.0, numpy.percentile(stats, p))
  p = (alpha+((1.0-alpha)/2.0)) * 100
  upper = min(1.0, numpy.percentile(stats, p))
  #print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
  return alpha, lower, upper
