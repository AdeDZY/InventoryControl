import cPickle as pickle
from util_conv import *


def main():
    [feat_mat1, rewards, feat_mat2] = pickle.load(open('features.pkl'))
    P = space_projection_mat(feat_mat1)
    gamma = 0.9
    d = np.dot(P, rewards)
    C = np.dot(P, feat_mat2) * gamma - feat_mat1
    eig, eigv = np.linalg.eig(C.T.dot(C))
    eig_max = eig.max()

    iternum = 20000
    mu = 10
    tau = 0.9 / eig_max
    epsilon = 0.1

    datanum, featnum = feat_mat1.shape
    beta = np.random.normal(size=featnum)
    v = np.random.normal(size=datanum)
    beta_best = None
    min_obj = 99999999

    # inexact admm
    for idx in range(iternum):
        c = d + C.dot(beta) - v * mu
        alpha = l2_ball_projection(c, epsilon)
        beta_grad = C.T.dot(C.dot(beta) + d - alpha - v * mu)
        beta_grad_update = beta - beta_grad * tau
        beta = l1_proximal_map(beta_grad_update, tau * mu)
        v = v - (C.dot(beta) + d - alpha) / mu

        obj = np.linalg.norm(np.dot(C, beta) + d)
        if obj < min_obj:
            min_obj = obj
            beta_best = beta

        if idx % 100 == 0:
            print obj

    pickle.dump(beta_best, open('beta_admm.pkl', 'wb'))


if __name__ == '__main__':
    main()
