import numpy as np
import random
import math
import cPickle as pickle


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
            reward -= action * self.buy_price - self.order_overhead
            reward += 0.2 * math.pow(self.current_state, 1.4)  # unsold units storage fee
        else:
            reward += (action + prev_state) * self.sell_price  # sell all
            reward -= action * self.buy_price - self.order_overhead
            reward -= (customer_demand - prev_state - action) * self.custom_punish  # unsatisfied demand punishment
            # no unsold units

        return reward

    def print_history(self):
        print self.history
        

def rbf(x, m, v):
    return np.exp(-np.square((x-m)/v))


def main():
    ice = InventoryControlEnv()
    random.seed()
    history = []

    # generate history under fill-all policy
    nsample = 100  # experiments compares nsample = 100 and 500
    for idx in range(nsample):
        ice.reset()
        for jdx in range(10):  # each episode has 10 steps
            # print cwe.EstimateState()
            action = 20 - ice.current_state
            reward = ice.take_action(action)

        ice.print_history()
        history.append(ice.history)
        # print len(history)
    pickle.dump(history, open('history.pkl', 'wb'))

if __name__ == '__main__':
    main()

