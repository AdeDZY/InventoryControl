#!/usr/bin/env bash

# generate history. 20 independent runs for each sample size

for i in {1..20}
do
    python InventoryControlEnv.py ${i} 100 # sample = 100 * 10
done

for i in {21..40}
do
    python InventoryControlEnv.py ${i} 500 # sample = 500 * 10
done

# Generate features.
for i in {1..40}
do
    python feature_vi.py ${i}
done

# Run and evaluate lstd

for i in {1..40}
do
    python lstd.py ${i}
    python evaluation_vi.py ${i} 0 >> lstd_result.dat
done

# Run and evaluate proximal_acc
for i in {1..40}
do
    python l1_pbr_proximal_acc.py ${i} >> acc.log
    echo "\n" >> acc.log
    python evaluation_vi.py ${i} 1 >> proximal_acc_result.dat
done


# Run and evaluate admm
for i in {1..40}
do
    python l1_pbr_admm.py ${i} >> admm.log
    echo "\n" >> admm.log
    python evaluation_vi.py ${i} 2 >> admm_result.dat
done
