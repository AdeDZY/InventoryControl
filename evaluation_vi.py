import numpy as np
from feature_vi import selected_feat
import cPickle as pickle

true_value = pickle.load(open('fillall_value.pkl', 'r'))
estimated_value = {}

beta = pickle.load(open('beta.pkl', 'r'))
iternum = 1
for state in true_value:
    value = 0.0
    for iter in range(iternum):
        feat = selected_feat(state)
        value += np.dot(beta, feat.T)
    value /= iternum
    print value, true_value[state]
    estimated_value[state] = value

mse = 0
for state in true_value:
    mse += (true_value[state] - estimated_value[state]) ** 2
mse /= len(true_value)
print estimated_value
print mse
