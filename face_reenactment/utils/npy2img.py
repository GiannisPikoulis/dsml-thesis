import numpy as np
from PIL import Image
import sys, os

# Sample values in range 0-1
npy_samples = np.load(str(sys.argv[1]))
print(f'max. value: {np.max(npy_samples)}, min. value: {np.min(npy_samples)}')
outdir = '.'.join(str(sys.argv[1]).split('.')[:-1])
print(f'Source: {str(sys.argv[1])}')
print(f'Destination: {outdir}')
os.makedirs(os.path.join(outdir), exist_ok=True)

if len(npy_samples.shape) == 5:
    E, N, H, W, C = npy_samples.shape
    npy_samples = npy_samples.reshape((E*N, H, W, C))
    total = E*N
else:
    N, H, W, C = npy_samples.shape
    total = N

print(f'Saving .npy file {str(sys.argv[1])} to directory {outdir}')
for i in range(total):
    print(f'Image index: {i}')
    im = Image.fromarray((npy_samples[i] * 255).astype(np.uint8))
    im.save(os.path.join(outdir, f'{i}.jpeg'))
       