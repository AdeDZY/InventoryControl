import numpy as np
import cPickle as pickle


def rbf(x, m, v):
    return np.exp(-np.square((x - m) / v))


def rele_feature_func(state):
    rel_feat = np.zeros((1, 21))
    rel_feat[0, 0] = state
    for fidx in range(0, 20):
        rel_feat[0, fidx + 1] = rbf(state, fidx * 1, 1)

    return rel_feat


def feature_func1(state):
    noise_feat_num = 1000
    irre_feat = np.random.normal(scale=20, size=(1, noise_feat_num))
    rel_feat = rele_feature_func(state)
    feat = np.concatenate((rel_feat, irre_feat), axis=1)

    return feat


def selected_feat(state):
    return feature_func1(state)


def main():
    history = pickle.load(open('history.pkl'))

    feat_total = None
    feat_total2 = None
    reward_total = None

    for idx in range(len(history)):
        samp_seq = history[idx]
        seq_size = len(samp_seq)
        if seq_size == 0:
            continue
        rewards = np.zeros(seq_size)
        actions = np.zeros(seq_size)

        for jdx in range(seq_size):
            state = samp_seq[jdx][0]
            state2 = samp_seq[jdx][3]
            rewards[jdx] = samp_seq[jdx][2]

            feat = selected_feat(state)
            feat2 = selected_feat(state2)

            if feat_total is None:
                feat_total = feat
                feat_total2 = feat2
            else:
                feat_total = np.concatenate((feat_total, feat), axis=0)
                feat_total2 = np.concatenate((feat_total2, feat2), axis=0)
        if reward_total is None:
            reward_total = rewards
        else:
            reward_total = np.concatenate((reward_total, rewards), axis=0)

    pickle.dump([feat_total, reward_total, feat_total2], open('features.pkl', 'wb'))


if __name__ == '__main__':
    main()
