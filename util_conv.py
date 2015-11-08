import numpy as np


def space_projection_mat(Phi):
    PhiT = Phi.T
    PhiTPhi_inv = np.linalg.pinv(np.dot(PhiT, Phi))
    P = np.dot(Phi, np.dot(PhiTPhi_inv, PhiT))
    return P


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
            #print "lala"
    return X_map


def l2_ball_projection(X, epsilon):
    if np.linalg.norm(X) <= epsilon:
        return X
    else:
        return X * epsilon / np.linalg.norm(X)
