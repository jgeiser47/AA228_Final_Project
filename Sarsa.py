#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:01:04 2020

@author: tristan
"""

import os
import pandas as pd
import numpy as np

input_file = "random_policy_runs.csv"
df = pd.read_csv(input_file)

s_data = df['s']
a_data = df['a']
r_data = df['r']
sp_data = df['sp']

gam = 0.5
S = 21
A = 2
alpha = 0.01
maxIters = 1

# Initialize action value function to all zeros
Q = np.zeros((S, A))

for i in range(maxIters):
    print(i)
    for k in range(len(df)-1):
        s = s_data[k]
        a = a_data[k]
        r = r_data[k]
        sp = sp_data[k]
        ap = a_data[k+1]

        Q[s, a] = Q[s, a] + alpha*(r + gam*Q[sp, ap] - Q[s, a]) #Sarsa
        #Q[s, a] = Q[s, a] + alpha*(r + gam*max(Q[sp, :]) - Q[s, a]) #Q-learn


policy = np.argmax(Q, axis=1)

# Get output file name and path
output_dir = os.getcwd()
output_file = os.path.join(output_dir, 'Sarsa_Policy.policy')

# Open output file
DF = open(output_file, 'w')

# Iterate through each value in policy, writing to output file
for i in range(S):
    DF.write(f'{policy[i]}\n')

# Close output file
DF.close()
