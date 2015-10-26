import sys
import numpy as np
import cPickle as pickle
from util_conv import *

class LSTD(object):

    def __init__(self,feat_mat1,rewards,feat_mat2,gamma=0.9):
        self.gamma = gamma
        self.feat_mat1 = feat_mat1
        self.feat_mat2 = feat_mat2
        self.rewards = rewards

        self.datanum,self.featnum = feat_mat1.shape
        
    def lstd(self):
        beta = self.l2_lstd(0)
        return beta

    def l2_lstd(self,ld):
        b = self.feat_mat1.T.dot(self.rewards)
        A = (self.feat_mat1.T.dot(self.feat_mat1)-
            self.feat_mat1.T.dot(self.feat_mat2)*self.gamma)
        A = A + ld*np.identity(self.featnum)
        beta = np.linalg.solve(A, b)
        self.beta = beta
        return beta

    def pbr(self):
        P = SpaceProjectionMat(self.feat_mat1)
        d = np.dot(P,self.rewards)
        C = np.dot(P,self.feat_mat2)*self.gamma-self.feat_mat1
        obj = np.linalg.norm(np.dot(C,self.beta)+d)
        return obj


def main():
    [feat_mat1, rewards, feat_mat2] = pickle.load(open('features.pkl'))
    lstd = LSTD(feat_mat1,rewards,feat_mat2)
    beta = lstd.lstd()
    #print beta
    print lstd.pbr()
    pickle.dump(beta,open('beta.pkl','wb'))

if __name__=='__main__':
    main()
