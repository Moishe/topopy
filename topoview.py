import os
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import show

fn = 'N40W106.hgt'

siz = os.path.getsize(fn)
dim = int(math.sqrt(siz/2))

assert dim*dim*2 == siz, 'Invalid file size'

data = np.fromfile(fn, np.dtype('>i2'), dim*dim).reshape((dim, dim))

fig = plt.gcf()
plt.axis('off')
plt.contour(range(0, dim), range(0, dim), data, levels=50, alpha = 0.8, linewidths = 0.1, colors='black')
plt.savefig('contour-image.svg', format='svg', dpi=1200)
show()