import numpy as np
import cPickle as pickle
from InventoryControlEnv import *


def main():
    ice = InventoryControlEnv()
    gamma = 0.9
    state_value = {}
    n_states = 100
    for idx in range(n_states):
        state = random.uniform(0, 20)
        state_value[state] = -1

    # For these states, use monte-carlo rollouts to compute function values
    # For each states, run w times, each run h steps
    w = 2000
    h = 200
    print(len(state_value))
    for state in state_value:
        v = 0.0
        for wdx in range(w):
            ice.set_initial_state(state)
            discount = 1
            total_reward = 0

            for hdx in range(h):
                action = ice.max_size - ice.current_state
                reward = ice.take_action(action)
                total_reward += reward * discount
                discount *= gamma

            v += total_reward

        v /= w
        print "\n\n"
        print v
        state_value[state] = v

    pickle.dump(state_value, open('fillall_value.pkl','wb'))

if __name__ == "__main__":
    main()
