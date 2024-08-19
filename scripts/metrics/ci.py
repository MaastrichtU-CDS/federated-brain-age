# Calculate the confidence intervals

import numpy as np

samples = 1000
sig_level = 0.05
# List with the age gap for each sample
labels = []
bootstrapped = []
labels = np.array(list(labels.values()))
while len(bootstrapped) < samples:
    indices = np.random.randint(0, len(labels), len(labels))
    bootstrapped.append(np.average(np.abs(labels[indices])))

print(np.sort(bootstrapped)[int(samples*sig_level/2) - 1])
print(np.sort(bootstrapped)[int(samples*(1-sig_level/2)) - 1])
