import numpy as np
from feature_vi import selected_feat
import cPickle as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("runid", help="run id, such as 1, 2..")
parser.add_argument("model", type=int, help="0:lst, 1:acc_proximal, 2:admm")
args = parser.parse_args()

true_value = pickle.load(open('fillall_value.pkl', 'r'))
estimated_value = {}

if args.model == 0:
    beta = pickle.load(open('beta{0}.pkl'.format(args.runid), 'r'))
elif args.model == 1:
    beta = pickle.load(open('beta_acc_prox{0}.pkl'.format(args.runid), 'r'))
elif args.model == 2:
    beta = pickle.load(open('beta_admm{0}.pkl'.format(args.runid), 'r'))
else:
    print "Error! model must be in 0, 1, 2!"
    exit(-1)

print np.count_nonzero(beta)
print beta[0:25]
iternum = 100

mse = 0.0
for iter in range(iternum):
    for state in true_value:
        feat = selected_feat(state)
        value = np.dot(beta, feat.T)
        mse += (true_value[state] - value) ** 2
        # print true_value[state], value
mse /= len(true_value) * iternum
print mse

