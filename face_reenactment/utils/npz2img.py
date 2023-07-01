import numpy as np
from PIL import Image
import sys, os

# Sample values in range 0-1
npz_samples = np.load(str(sys.argv[1]))['arr_0']
print(f'max. value: {np.max(npz_samples)}, min. value: {np.min(npz_samples)}')
outdir = '.'.join(str(sys.argv[1]).split('.')[:-1])
print(f'Source: {str(sys.argv[1])}')
print(f'Destination: {outdir}')
os.makedirs(os.path.join(outdir), exist_ok=True)

if len(npz_samples.shape) == 5:
    E, N, H, W, C = npz_samples.shape
    npz_samples = npz_samples.reshape((E*N, H, W, C))
    total = E*N
else:
    N, H, W, C = npz_samples.shape
    total = N

print(f'Saving .npz file {str(sys.argv[1])} to directory {outdir}')
for i in range(total):
    print(f'Image index: {i}')
    im = Image.fromarray(npz_samples[i])
    im.save(os.path.join(outdir, f'{i}.jpeg'))