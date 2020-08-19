
from load import load_images
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, PaddedBox
import numpy as np
from pathlib import Path

def get_pca(data, n_components=2):
    return PCA(n_components=n_components).fit_transform(data)

def PCA_ImageSpaceVisualization(data):
    figure = plt.figure()
    ax = plt.gca()

    def scatter_image(x, y, array, zoom=1):
        im = OffsetImage(array, zoom=zoom)
        x, y = np.atleast_1d(x, y)

        artists = []
        for x, y in zip(x, y):
            ab = AnnotationBbox(im, (x, y), xycoords='data')
            artists.append(ax.add_artist(ab))

        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists

    projected = get_pca(data)

    images = [
        Image.fromarray(image.reshape(100, 100, 3), 'RGB')
        for image in data
    ]

    for (x, y), image in zip(projected, images):
        scatter_image(x, y, image, zoom=0.5)
        plt.scatter(x, y)

    plt.title("Principal Component Analysis of Seasonal Images")
    plt.xlabel('Component #1')
    plt.ylabel('Component #2')
    plt.xlim(
            projected[:,0].min() * 1.5,
            projected[:,0].max() * 1.5
    )
    plt.ylim(
            projected[:,1].min() * 1.5,
            projected[:,1].max() * 1.5
    )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data, _ = load_images("./img")
    PCA_ImageSpaceVisualization(data)
