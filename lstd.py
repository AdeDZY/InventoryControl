import cPickle as pickle
from util_conv import *
import argparse


class LSTD(object):

    def __init__(self, feat_mat1, rewards, feat_mat2, gamma=0.9):
        self.gamma = gamma
        self.feat_mat1 = feat_mat1
        self.feat_mat2 = feat_mat2
        self.rewards = rewards

        self.datanum, self.featnum = feat_mat1.shape
        
    def lstd(self):
        beta = self.l2_lstd(0)
        return beta

    def l2_lstd(self, ld):
        b = self.feat_mat1.T.dot(self.rewards)
        A = (self.feat_mat1.T.dot(self.feat_mat1) -
             self.feat_mat1.T.dot(self.feat_mat2) * self.gamma)
        A = A + ld * np.identity(self.featnum)
        beta = np.linalg.solve(A, b)
        self.beta = beta
        return beta

    def pbr(self):
        P = space_projection_mat(self.feat_mat1)
        d = np.dot(P, self.rewards)
        C = np.dot(P, self.feat_mat2) * self.gamma - self.feat_mat1
        obj = np.linalg.norm(np.dot(C, self.beta) + d)
        return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="run id, such as 1, 2..")
    args = parser.parse_args()

    [feat_mat1, rewards, feat_mat2] = pickle.load(open('features{0}.pkl'.format(args.runid)))
    lstd = LSTD(feat_mat1, rewards, feat_mat2)
    beta = lstd.lstd()
    print lstd.pbr()
    print beta
    pickle.dump(beta, open('beta{0}.pkl'.format(args.runid), 'wb'))

if __name__ == '__main__':
    main()
