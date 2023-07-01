import numpy as np
import os, sys

# Sample values in range 0-1
npy_samples = np.load(str(sys.argv[1]))
outdir = '.'.join(str(sys.argv[1]).split('.')[:-1])
npz_path = outdir + '.npz'

print(f'SRC PATH: {str(sys.argv[1])}')
print(f'DST PATH: {npz_path}')

if len(npy_samples.shape) == 5:
    E, N, H, W, C = npy_samples.shape
    npy_samples = npy_samples.reshape((E*N, H, W, C))

print(f'max. value: {np.max(npy_samples)}, min. value: {np.min(npy_samples)}')
npy_samples = (npy_samples * 255).astype(np.uint8)
print(f'max. value: {np.max(npy_samples)}, min. value: {np.min(npy_samples)}')
print(f'Data shape: {npy_samples.shape}')
np.savez(npz_path, npy_samples)