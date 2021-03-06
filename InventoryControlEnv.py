import numpy as np
import random
import math
import cPickle as pickle
import argparse


class InventoryControlEnv(object):

    def __init__(self):
        random.seed()
        self.max_size = 20
        self.buy_price = 2
        self.sell_price = 2.5
        self.order_overhead = 2
        self.custom_punish = 0.5
        self.current_state = random.uniform(0, self.max_size)
        self.history = []

    def reset(self):
        self.current_state = random.uniform(0, self.max_size)
        self.history = []

    def set_initial_state(self, init_state):
        self.current_state = init_state
        self.history = []

    def take_action(self, action):
        prev_state = self.current_state
        customer_demand = random.uniform(0, 24)

        self.current_state = max(prev_state + action - customer_demand, 0)
        reward = self.instant_reward(action, prev_state, customer_demand)

        self.history.append((prev_state, action, reward, self.current_state))
        return reward

    def instant_reward(self, action, prev_state, customer_demand):
        reward = 0
        if prev_state + action >= customer_demand:
            reward += customer_demand * self.sell_price
            reward -= action * self.buy_price + self.order_overhead
            reward -= 0.2 * math.pow(self.current_state, 1.4)  # unsold units storage fee
        else:
            reward += (action + prev_state) * self.sell_price  # sell all
            reward -= action * self.buy_price + self.order_overhead
            reward -= (customer_demand - prev_state - action) * self.custom_punish  # unsatisfied demand punishment
            # no unsold units

        return reward

    def print_history(self):
        print self.history
        

def rbf(x, m, v):
    return np.exp(-np.square((x-m)/v))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runid", help="run id, such as 1, 2..")
    parser.add_argument("nsample", type=int, help="# of outer iterations")
    args = parser.parse_args()

    ice = InventoryControlEnv()
    random.seed()
    history = []

    # generate history under fill-all policy
    nsample = args.nsample  # experiments compares nsample = 100 and 500
    for idx in range(nsample):
        ice.reset()
        for jdx in range(20):  # each episode has 10 steps
            action = ice.max_size - ice.current_state
            reward = ice.take_action(action)
        # ice.print_history()
        history.append(ice.history)
    pickle.dump(history, open('history{0}.pkl'.format(args.runid), 'wb'))

if __name__ == '__main__':
    main()

