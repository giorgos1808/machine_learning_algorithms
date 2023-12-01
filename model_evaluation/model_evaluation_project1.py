# HOMEWORK 7 - Model Evaluation
# Friedman & Nemenyi test
# Tsakiris Giorgos

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

algo_performance = pd.read_csv('algo_performance.csv')

friedman_res = friedmanchisquare(algo_performance['C4.5'], algo_performance['1-NN'], algo_performance['NaiveBayes'], algo_performance['Kernel'], algo_performance['CN2'])
print('Friedman test')
print('Test Statistic: %s, p-value: %s' %(str(friedman_res[0]), str(friedman_res[1])))

data = np.array([algo_performance['C4.5'], algo_performance['1-NN'], algo_performance['NaiveBayes'], algo_performance['Kernel'], algo_performance['CN2']])

print('\nNemenyi test (posthoc_nemenyi)')
print('Dictionary: C4.5 = 1, 1-NN = 2, NaiveBayes = 3, Kernel = 4, CN2 = 5')
nemenyi_res = sp.posthoc_nemenyi(data)
print(nemenyi_res)

print('\nNemenyi test (posthoc_nemenyi_friedman)')
print('Dictionary: C4.5 = 0, 1-NN = 1, NaiveBayes = 2, Kernel = 3, CN2 = 4')
nemenyi_res = sp.posthoc_nemenyi_friedman(data.T)
print(nemenyi_res)
