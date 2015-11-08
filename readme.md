# Inventory Control
## Environment
- InventoryControlEnv.py: simulator of inventory control, for data collection
- feature_vi.py: feature for value function iteration. 
  - 1 bias (feat[0] = 1)
  - 1 state (feat[1])
  - 20 rbf (m = 0,1, .. 20) 
  - 1000 noisy features
- feature_pi.py: feature for policy iteration. Todo

## Methods
- lstd: least square temporal difference learning (baseline)
- l1_pbr_proximal_acc.py: use accelerated proximal gradient descent to solve l1 regularized pbr
- l1_pbr_admm.py: use modified admm to solve (pbr as a constraint)
- linesearch: todo
