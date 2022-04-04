import json
import os
import math
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import show

def xlat_coord(coord, dx0, dy0):
    x = int((coord[0] - dx0) * 3600)
    y = int((coord[1] - dy0) * 3600)
    return (x,y)

try:
    data = np.load('data.npy')
    offsets = np.load('offsets.npy')
    pixel_coordinates = np.load('pixel-coords.npy')
except OSError:
    data = None

if data is None:
    fns = ['N40W106.hgt', 'N39W106.hgt']

    data = None
    for fn in fns:
        siz = os.path.getsize(fn)
        dim = int(math.sqrt(siz/2))
        print(dim)

        assert dim*dim*2 == siz, 'Invalid file size'

        quad = np.fromfile(fn, np.dtype('>i2'), dim*dim).reshape((dim, dim))
        if data is not None:
            data = np.concatenate((data, quad), axis=0)
        else:
            data = quad

    f = open('County_Boundary.geojson')
    boundary = json.load(f)
    coordinates = boundary['features'][0]['geometry']['coordinates'][0][0]
    boundary_path = matplotlib.path.Path(coordinates)
    bbox = boundary_path.get_extents()
    x_count = int((bbox.x1 - bbox.x0) * 3600)
    y_count = int((bbox.y1 - bbox.y0) * 3600)
    xl = np.linspace(bbox.x0, bbox.x1, x_count, endpoint=True)
    yl = np.linspace(bbox.y0, bbox.y1, y_count, endpoint=True)
    X,Y = np.meshgrid(xl, yl)

    dx0 = int((bbox.x0 - -106) * 3600)
    dx1 = dx0 + x_count
    dy0 = int((bbox.y0 - 39) * 3600)
    dy1 = dy0 + y_count
    data = data[dy0:dy1, dx0:dx1]

    pixel_coordinates = [xlat_coord(coord, bbox.x0, bbox.y0) for coord in coordinates]

    coordinate_grid = np.array([X,Y])
    d = np.dstack((coordinate_grid[0], coordinate_grid[1]))
    coords = [el for sublist in d for el in sublist]
    contains_points = boundary_path.contains_points(coords)
    normalized_contains = np.reshape(contains_points, (y_count, x_count))

    mn = np.min(data)
    mx = np.max(data)
    print('Lowest/highest elevation in Boulder County: %d/%d' % (mn, mx))

    bounded_data = data * normalized_contains

    #data = np.where(data == 0, mn, data)

    np.save('data', data)
    np.save('bounded-data', bounded_data)

    offsets = np.array([dx0, dy0])
    np.save('offsets', offsets)
    np.save('pixel-coords', pixel_coordinates)

fig, ax = plt.subplots()
plt.axis('off')
plt.tight_layout()
plt.gcf().set_size_inches(10, 8)

mn = 3000 #1500
mx = 4200 #4200
step = 100
l = [mn + x * step for x in range(0, int(np.floor((mx - mn) / step) + 1))]
print("%d contours" % len(l))

pixel_path = matplotlib.path.Path(pixel_coordinates)
clip_path = patches.PathPatch(pixel_path, color=None, fill=False, visible=False)
plt.gca().add_patch(clip_path)

cont = plt.contour(range(0, data.shape[1]), range(0, data.shape[0]), data, l, alpha = 0.5, linewidths = 0.1, colors='black')
for c in cont.collections:
    c.set_clip_path(clip_path)

home = (-105.24763 - -106)

plt.savefig('contour-image-mountain.svg', format='svg', dpi=1200)
show()