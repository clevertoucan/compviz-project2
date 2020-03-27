from roipoly import RoiPoly as poly
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.image as img
from matplotlib.path import Path as MplPath
import glob
import csv


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

for i in train_images:
    filename = i[14:-4]
    arr = img.imread(i)
    plt.imshow(arr)
    roi = poly(color='r')
    save_roi("./masks/" + filename + ".csv", roi)

csv_path = r'./masks/'
masks = glob.glob(csv_path + "/*.csv")

for m in masks:
    filename = m[12:-4]
    roi = load_roi(m)
    arr = img.imread(img_path + filename + ".jpg")
    mask = get_mask(roi, arr)
    plt.imsave("./masks/" + filename + ".png", mask)


