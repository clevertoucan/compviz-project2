import roi
import matplotlib.pyplot as plt
import numpy as np
import colorseg
import draw_ellipsoid

roi_data = roi.load_all_roi_data()
m = roi_data[0]
d = m[0][m[1].get_mask(m[0])]
mean, cov, probs = colorseg.single_gauss(d)
fig = plt.figure(figsize=2.5 * plt.figaspect(1))
ax = fig.gca(projection='3d')
r = np.asarray(m[0][:, :, 0]).flatten()
g = np.asarray(m[0][:, :, 1]).flatten()
b = np.asarray(m[0][:, :, 2]).flatten()
p = list(map(lambda x, y, z: [x/255, y/255, z/255], r, g, b))
ax.scatter(r, g, b, s=10, facecolors=p)
draw_ellipsoid.confidence_ellipsoid3(mean, cov, ax)
plt.show()

stats = []
for m in roi_data:
    d = m[0][m[1].get_mask(m[0])]
    (mean, cov, probs) = colorseg.single_gauss(d)
    stats.append((mean, cov, probs))
stats
