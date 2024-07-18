import cv2
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Set the seed for reproducibility
np.random.seed(0)
rng = np.random.default_rng()


def rand_points(n_points=10) -> np.ndarray[np.float64]:
    # Generate N= n_points random points in the range [0, n_points * 10]
    return rng.uniform(low=0, high=n_points * 10, size=(n_points, 2))


def setup_plot():
    # Set up a 3000x1000 pixel figure
    fig = plt.figure(figsize=(30, 10))
    axs = fig.subplots(nrows=1, ncols=3)

    # Turn off axis and labels
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axs


def get_scatter_plot(points: np.ndarray[np.float64], ax: plt.Axes):
    # Plot the points as black dots
    ax.scatter(points[:, 0], points[:, 1], s=1, c="k")
    ax.scatter(points[:, 0], points[:, 1], s=1, c="k")
    ax.set_title(f"{len(points)} Random Points", fontsize=14)
    ax.invert_yaxis()

    # Save the figure to a buffer
    img = io.BytesIO()
    extent = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    ax.figure.savefig(img, bbox_inches=extent, pad_inches=0, format="png")

    # Set file pointer to the beginning of the buffer
    _ = img.seek(0)


def main():
    pass


if __name__ == "__main__":
    main()
