# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours, approximate_polygon
from skimage.io import imread

from image_reader import BinaryImageFromFile

if __name__ == "__main__":
    img_path = "data/rect2.jpg"
    img = BinaryImageFromFile(img_path)
    contours = find_contours(img.array)

    for tol in [0,
                0.1, 0.5, 1, 5, 10, 20, 50, 100
                ]:
        plt.subplot(1, 2, 2)
        plt.imshow(1 - img.array, cmap=plt.cm.gray)
        for contour in contours:
            contour = approximate_polygon(contour, tolerance=tol)
            plt.plot(contour[..., 1], contour[..., 0])
        plt.title('ground truth')
        plt.subplot(1, 2, 1)
        plt.imshow(img.array * np.nan, cmap=plt.cm.gray)
        for contour in contours:
            contour = approximate_polygon(contour, tolerance=tol)
            plt.plot(contour[..., 1], contour[..., 0])
        plt.title(f"RDP approximation (tolerence = {tol})")
        plt.show()
    pass
