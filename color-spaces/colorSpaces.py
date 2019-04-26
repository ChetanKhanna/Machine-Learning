import cv2
import matplotlib.pyplot as plt
import numpy as np
# imports for plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# Reads in defualt BGR coloring -- OpenCV by defaults reads in BGR
nemo = cv2.imread('./images/nemo0.jpg')
# Changing to RGB coloring
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
# plotting RGB scatter plot
r, g, b = cv2.split(nemo)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection='3d')
# setting up pixel colors
pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
# displaying plot
axis.scatter(
    r.flatten(), g.flatten(), b.flatten(),
    facecolors=pixel_colors, marker='.')
axis.set_xlabel('Red')
axis.set_ylabel('Green')
axis.set_zlabel('Blue')
# uncomment to show
# plt.show()
# Changing to hsv for better segmentation
nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
# plotting HSV scatter plot
h, s, v = cv2.split(nemo_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection='3d')
# setting up pixel colors
# facecolor must be in RGB so using the same pixel_colors list
axis.scatter(
    h.flatten(), s.flatten(), v.flatten(),
    facecolors=pixel_colors, marker='.')
axis.set_xlabel('Hue')
axis.set_ylabel('Saturation')
axis.set_zlabel('Value')
# uncomment to show
# plt.show()
# displaying image
# plt.imshow(nemo)
# uncomment to show
# plt.show()

# setting bounds for orange color
light_orange = (1, 190, 200)
dark_orange = (18, 255, 255)
# making a mask from hsv convetted image
mask = cv2.inRange(nemo_hsv, light_orange, dark_orange)
# extracting matched image from RGB image
result = cv2.bitwise_and(nemo, nemo, mask=mask)
# displaying original and result
# plt.subplot(1, 2, 1)
# plt.imshow(mask, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(result)
# plt.show()
# selecting bounds for white mask
ligh_white = (0, 0, 200)
dark_white = (145, 60, 255)
# making a white mask
mask_white = cv2.inRange(nemo_hsv, ligh_white, dark_white)
result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)
# uncoment to see the result of white mask
# plt.subplot(1, 2, 1)
# plt.imshow(mask_white, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(result_white)
# plt.show()
# combining the two masks
final_mask = mask + mask_white
final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
plt.subplot(1, 2, 1)
plt.imshow(final_mask)
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()
