import numpy as np
from pathlib import Path


path = Path("./out.flo")

with path.open(mode='r') as flo:
  tag = np.fromfile(flo, np.float32, count=1)[0]
  width = np.fromfile(flo, np.int32, count=1)[0]
  height = np.fromfile(flo, np.int32, count=1)[0]
  print('tag', tag, 'width', width, 'height', height)
  nbands = 2
  tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
  flow = np.resize(tmp, (int(height), int(width), int(nbands)))
