import numpy as np
import cPickle as pickle
from util_conv import *


def main():
    iternum = 20000
    [feat_mat1, rewards, feat_mat2] = pickle.load(open('features.pkl'))
    P = space_projection_mat(feat_mat1)
    gamma = 0.9
    ld = 0.1
    lr = 0.0000001
    d = np.dot(P, rewards)
    C = np.dot(P, feat_mat2) * gamma - feat_mat1
    datanum, featnum = feat_mat1.shape
    beta = np.random.normal(size=featnum)
    beta_t_1 = beta
    beta_best = None
    min_obj = 99999999

    for idx in range(iternum):
        beta_v = beta + (beta - beta_t_1) * (idx - 2) / (idx + 1)
        beta_t_1 = beta
        beta_grad = np.dot(C.T, np.dot(C, beta) + d)
        beta_grad_update = beta_v - beta_grad * lr
        beta = l1_proximal_map(beta_grad_update, ld * lr)
        obj = np.linalg.norm(np.dot(C, beta) + d) + ld * np.linalg.norm(beta, 1)
        if obj < min_obj:
            min_obj = obj
            beta_best = beta

        if idx % 100 == 0:
            print obj

    # print beta
    print min_obj
    print beta_best
    pickle.dump(beta_best, open('beta_acc_prox.pkl', 'wb'))


if __name__ == '__main__':
    main()
