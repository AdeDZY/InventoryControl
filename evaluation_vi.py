import numpy as np
from feature_vi import selected_feat
import cPickle as pickle

true_value = pickle.load(open('fillall_value.pkl', 'r'))
estimated_value = {}

beta = pickle.load(open('beta.pkl', 'r')) # !! change here !!

print np.count_nonzero(beta)
print beta[0:25]
iternum = 100

mse = 0.0
for iter in range(iternum):
    for state in true_value:
        feat = selected_feat(state)
        value = np.dot(beta, feat.T)
        mse += (true_value[state] - value) ** 2
        #print true_value[state], value
mse /= len(true_value) * iternum
print mse

