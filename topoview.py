import json
import os
import math
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import show

try:
    data = np.load('data.npy')
    offsets = np.load('coords.npy')
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
    #data = np.full((y_count, x_count), 1)

    coordinate_grid = np.array([X,Y])
    d = np.dstack((coordinate_grid[0], coordinate_grid[1]))
    coords = [el for sublist in d for el in sublist]
    contains_points = boundary_path.contains_points(coords)
    normalized_contains = np.reshape(contains_points, (y_count, x_count))
    print(normalized_contains.shape)
    print(data.shape)
    #print(boundary_path.contains_point((-106, -40.1)))

    data = data * normalized_contains
    np.save('data', data)

    offsets = np.array([dx0, dy0])
    np.save('offsets', offsets)

fig = plt.gcf()
#plt.axis('off')
#by_el = np.fromfunction(lambda x,y: xlat_el(x, y, data, boundary_path), (dim, dim))
plt.contour(range(0, data.shape[1]), range(0, data.shape[0]), data, levels=100, alpha = 0.8, linewidths = 0.1, colors='black')
x = (-105.24763 - -106) * data.shape[1] + offsets[0]
y = data.shape[0] - (40.05008 - 39) * data.shape[0] + offsets[0]
plt.scatter([x], [y], label='Home', s=50)
plt.savefig('contour-image.svg', format='svg', dpi=1200)
show()