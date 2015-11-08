import numpy as np
import cPickle as pickle
from util_conv import *


def l1_proximal_map(X, t):
    X_map = X
    dim = X_map.shape[0]
    for idx in range(dim):
        if X_map[idx] > t:
            X_map[idx] -= t
        elif X_map[idx] < -t:
            X_map[idx] += t
        else:
            X_map[idx] = 0
    return X_map


def main():
    iternum = 50000
    [feat_mat1, rewards, feat_mat2] = pickle.load(open('features.pkl'))
    P = space_projection_mat(feat_mat1)
    gamma = 0.9
    ld = 1
    lr = 0.000001
    d = np.dot(P, rewards)
    C = np.dot(P, feat_mat2) * gamma - feat_mat1
    datanum, featnum = feat_mat1.shape
    beta = np.zeros(featnum)

    for idx in range(iternum):
        beta_grad = np.dot(C.T, np.dot(C, beta) + d)
        beta_grad_update = beta - beta_grad * lr
        beta = l1_proximal_map(beta_grad_update, ld * lr)
        obj = np.linalg.norm(np.dot(C, beta) + d) + ld * np.linalg.norm(beta, 1)
        if idx % 100 == 0:
            print obj
            print np.count_nonzero(beta)

    print beta
    print obj
    pickle.dump(beta, open('beta_prox.pkl', 'wb'))


if __name__ == '__main__':
    main()
