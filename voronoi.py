import cv2
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Set the seed for reproducibility
np.random.seed(0)
rng = np.random.default_rng()


def rand_points(n_points=10):
    return rng.uniform(low=0, high=n_points * 10, size=(n_points, 2))


def setup_plot():
    fig = plt.figure(figsize=(30, 10))
    axs = fig.subplots(nrows=1, ncols=3)

    # Turn off axis and labels
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axs


def gen_image(points, axs):
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=axs[0])
    axs[0].set_title("Voronoi Diagram")
