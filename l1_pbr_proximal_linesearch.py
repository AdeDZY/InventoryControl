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
    ld = 100000
    d = np.dot(P, rewards)
    C = np.dot(P, feat_mat2) * gamma - feat_mat1
    datanum, featnum = feat_mat1.shape
    beta = np.zeros(featnum)

    for idx in range(iternum):
        lr = 1.0
        beta_grad = np.dot(C.T, np.dot(C, beta) + d)
        obj_g = np.linalg.norm(np.dot(C, beta) + d)
        while True:
            beta_grad_update = beta - beta_grad * lr
            beta_new = l1_proximal_map(beta_grad_update, ld * lr)
            print beta_new
            beta_G = (1 / lr) * (beta - beta_new)
            obj_g_new = np.linalg.norm(np.dot(C, beta_new) + d)
            diff = obj_g_new - obj_g
            print "diff " + str(diff)
            print "1 " + str(-lr * beta_grad_update.dot(beta_G))
            print "2 " + str(lr / 2 * np.dot(beta_G, beta_G))
            print "3 " + str(np.dot(beta_G, beta_G))
            if diff <= -lr * beta_grad_update.dot(beta_G) + lr / 2 * np.dot(beta_G, beta_G):
                beta = beta_new
                print lr
                break
            else:
                lr *= 0.5
                print lr

        if idx % 100 == 0:
            obj = np.linalg.norm(np.dot(C, beta) + d) + ld * np.linalg.norm(beta, 1)
            print obj
            print np.count_nonzero(beta)

    print beta
    print obj
    pickle.dump(beta, open('beta_prox.pkl', 'wb'))


if __name__ == '__main__':
    main()
