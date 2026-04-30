import numpy as np

file_1 = "results/sim_study/abc_samples_large_v2.npz"
file_2 = "results/sim_study/abc_samples_large_v2_second.npz"
file_merged = "results/sim_study/abc_samples_large_merged.npz"

d1 = np.load(file_1, allow_pickle=True)
d2 = np.load(file_2, allow_pickle=True)

abc_samples          = np.concatenate([d1['abc_samples'],          d2['abc_samples']],          axis=0)
abc_samples_normalized = np.concatenate([d1['abc_samples_normalized'], d2['abc_samples_normalized']], axis=0)
abc_map              = np.concatenate([d1['abc_map'],              d2['abc_map']],              axis=0)
abc_map_normalized   = np.concatenate([d1['abc_map_normalized'],   d2['abc_map_normalized']],   axis=0)
true                 = np.concatenate([d1['true'],                 d2['true']],                 axis=0)
true_normalized      = np.concatenate([d1['true_normalized'],      d2['true_normalized']],      axis=0)

N_train = len(abc_samples)
print(f"Merged {len(d1['abc_samples'])} + {len(d2['abc_samples'])} = {N_train} test points")

np.savez(file_merged, abc_samples=abc_samples, abc_samples_normalized=abc_samples_normalized,
         abc_map=abc_map, abc_map_normalized=abc_map_normalized,
         N_train=N_train, true=true, true_normalized=true_normalized)

print(f"Saved merged file to {file_merged}")
