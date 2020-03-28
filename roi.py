from roipoly import RoiPoly as poly
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.image as img
from matplotlib.path import Path as MplPath
import glob
import csv
from typing import Tuple, List


def get_mask(self, current_image):
    ny, nx, z = np.shape(current_image)
    poly_verts = ([(self.x[0], self.y[0])]
                  + list(zip(reversed(self.x), reversed(self.y))))
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    roi_path = MplPath(poly_verts)
    grid = roi_path.contains_points(points).reshape((ny, nx))
    return grid


def save_roi(filepath, roi):
    a = np.asarray([roi.x, roi.y])
    np.savetxt(filepath, a, delimiter=",")


def load_roi(csv_filepath):
    c = []
    reader = csv.reader(open(csv_filepath))
    for row in reader:
        r = []
        for n in row:
            n = float(n)
            r.append(n)
        c.append(r)
    x = c[0]
    y = c[1]
    roi = poly(show_fig=False)
    roi.completed = True
    roi.x = x
    roi.y = y
    return roi


img_path = r'./train_images/'
train_images = glob.glob(img_path + "/*.jpg")


def load_all_roi_data() -> List[Tuple[np.array, poly]]:
    csv_path = r'./roi_coords/'
    masks = glob.glob(csv_path + "/*.csv")
    roi_data = []
    for m in masks:
        filename = m[12:-4]
        roi_data.append((img.imread(img_path + filename + ".jpg"), load_roi(m)))
    return roi_data


