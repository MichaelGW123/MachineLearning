# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Importing the dataset
path = Path(__file__).parent / 'Ads_CTR_Optimisation.csv'
dataset = pd.read_csv(path)

# Implementing UCB
import math
N = 10000
d = 10
adsSelected = []
numberOfSelections = [0]*d
rewards = [0]*d
totalReward = 0

for n in range(0,N):
    ad = 0
    maxUCB = 0
    for i in range(0, d):
        if numberOfSelections[i]>0:
            averageReward = rewards[i]/numberOfSelections[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/numberOfSelections[i])
            upperBound = averageReward + delta_i
        else:
            upperBound = 1e400
        if (upperBound > maxUCB):
            maxUCB = upperBound
            ad = i
    adsSelected.append(ad)
    numberOfSelections[ad]+=1
    reward = dataset.values[n, ad]
    rewards[ad]+= reward
    totalReward+= reward

# Visualizing the results
plt.hist(adsSelected)
plt.title('Histogram of Ads Selected')
plt.xlabel('Ads')
plt.ylabel('Number of times each Ad was selected')
plt.show()