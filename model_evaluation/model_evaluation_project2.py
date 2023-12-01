# HOMEWORK 7 - Model Evaluation
# Pearson & Spearman correlation
# Tsakiris Giorgos

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

data = pd.read_csv('GeorgeData.csv')

step_value = []
for data_idx in range(len(data)):
    step_value.append(data.loc[data_idx][0])

step_value_split = []
for idx in range(len(step_value)):
    step_value_split.append(step_value[idx].split(';'))
step_value_array = np.float32(np.array(step_value_split))

steps = step_value_array[:, 0]
values = step_value_array[:, 1]

pearson_r, pearson_p = pearsonr(steps, values)
spearman_r, spearman_p = spearmanr(steps, values)

print("Pearson's correlation: Test Statistic: %s, p-value: %s" % (str(pearson_r), str(pearson_p)))
print("Spearman's correlation: Test Statistic: %s, p-value: %s" % (str(spearman_r), str(spearman_p)))

idx = np.argmax(values)
print('Max Reward: %f, at %d steps' %(np.max(values), int(steps[idx])))

plt.plot(steps, values)
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.show()
