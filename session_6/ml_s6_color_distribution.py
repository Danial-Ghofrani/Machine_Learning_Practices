import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("face.png", cv.IMREAD_GRAYSCALE)

print(img.shape)

print(img.min())
print(img.max())

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")

plt.subplot(1,2,2)
plt.hist(img.ravel(), bins=256)
plt.show( )