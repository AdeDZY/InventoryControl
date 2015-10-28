import numpy as np
import cPickle as pickle
import random


def rbf(x, m, v):
    return np.exp(-np.square((x-m)/v))


def random_policy():
    random.seed()
    if random.random() > 0.5:
        return 'right'
    else:
        return 'left'


def main():
    noise_feat_num = 1000
    history = pickle.load(open('history.pkl'))

    feat_total = None
    feat_total2 = None

    for idx in range(len(history)):
        samp_seq = history[idx]
        seq_size = len(samp_seq)
        if seq_size == 0:
            continue
        rewards = np.zeros(seq_size)
        actions = np.zeros(seq_size)
        dummy_feat = np.zeros((1, (6 + noise_feat_num)))

        for jdx in range(seq_size):
            state = samp_seq[jdx][0]
            state2 = samp_seq[jdx][3]
            rewards[jdx] = samp_seq[jdx][2]

            irre_feat = np.random.normal(size=(1, noise_feat_num))
            irre_feat2 = np.random.normal(size=(1, noise_feat_num))
            rel_feat = np.zeros((1, 11))
            rel_feat2 = np.zeros((1, 11))

            rel_feat[0, 0] = state
            rel_feat2[0, 0] = state2
            for fidx in range(0, 10):
                rel_feat[0, fidx + 1] = rbf(state, fidx * 4, 1)
                rel_feat2[0, fidx + 1] = rbf(state2, fidx * 4, 1)

            feat = np.concatenate((rel_feat, irre_feat), axis=1)
            feat2 = np.concatenate((rel_feat2, irre_feat2), axis=1)

            # TODO: policy feature
            # if samp_seq[jdx][1]=='right':
            #     actions[jdx] = 1
            #     feat = np.concatenate((dummyfeat, feat),axis = 1)
            # else:
            #     feat = np.concatenate((feat, dummyfeat),axis = 1)
            # if RandomPolicy()=='right':
            #     feat2 = np.concatenate((dummyfeat, feat2),axis = 1)
            # else:
            #     feat2 = np.concatenate((feat2, dummyfeat),axis = 1)
            #
            # if feat_total is None:
            #     feat_total = feat
            #     feat_total2 = feat2
            # else:
            #     feat_total = np.concatenate((feat_total, feat), axis = 0)
            #     feat_total2 = np.concatenate((feat_total2, feat2), axis = 0)

    print feat_total.shape
    print feat_total2.shape

if __name__ == "main":
    main()

