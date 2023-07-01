import numpy as np
from PIL import Image
import os, sys

input_dir = str(sys.argv[1])
out_dir = str(sys.argv[2])
os.makedirs(out_dir, exist_ok=True)
size = int(sys.argv[3])

cnt = 0
for file in os.listdir(input_dir):
    img = Image.open(os.path.join(input_dir, file))
    img = img.resize((size, size))
    img.save(os.path.join(out_dir, file))
    cnt += 1
    print(cnt)