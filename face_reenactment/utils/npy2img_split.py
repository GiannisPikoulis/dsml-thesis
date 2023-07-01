import numpy as np
from PIL import Image
import sys, os

# Sample values in range 0-1
npy_samples = np.load(str(sys.argv[1]))
outdir = '.'.join(str(sys.argv[1]).split('.')[:-1])
print(f'Source: {str(sys.argv[1])}')
print(f'Destination: {outdir}')
os.makedirs(os.path.join(outdir), exist_ok=True)

print(npy_samples.shape)
assert len(npy_samples.shape) == 5
E, N, H, W, C = npy_samples.shape

# print(f'Saving .npy file {str(sys.argv[1])} to directory {outdir}')
# for e in range(E):
#     print(f'Emotion {e}')
#     os.makedirs(os.path.join(outdir, str(e)), exist_ok=True)
#     for i in range(N):
#         print(f'Image index: {i}')
#         im = Image.fromarray((npy_samples[e][i] * 255).astype(np.uint8))
#         im.save(os.path.join(outdir, str(e), f'{i}.jpeg'))