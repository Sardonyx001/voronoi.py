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
        # axis corners mess up opencv countours and identify (0,0) as an extra
        # point so turn them off for now
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axs


def get_scatter_plot(points: np.ndarray[np.float64], ax: plt.Axes) -> io.BytesIO:
    # Plot the points as black dots
    ax.scatter(points[:, 0], points[:, 1], s=1, c="k")
    ax.scatter(points[:, 0], points[:, 1], s=1, c="k")
    ax.set_title(f"{len(points)} Random Points", fontsize=14)
    ax.invert_yaxis()

    # Save the figure to a buffer
    img = io.BytesIO()
    extent = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    ax.figure.savefig(img, bbox_inches=extent, format="png")

    # # Save the image to disk
    # with open("./images/random_points.png", "wb") as f:
    #     f.write(img.getbuffer())

    # Set file pointer to the beginning of the buffer
    _ = img.seek(0)
    return img


def extract_points(
    img: io.BytesIO,
) -> tuple[np.ndarray[np.float64], cv2.typing.MatLike]:
    # Read image from buffer
    image_cv = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get binary image (black points on white background)
    _, binary_image = cv2.threshold(image_cv, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the points
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the centers of the contours
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    return np.array(centers), image_cv


def plot_centers(
    centers: np.ndarray[np.float64], ax: plt.Axes, image_cv: cv2.typing.MatLike
) -> None:
    # Draw the centers on the image
    image_bgr = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
    for center in centers:
        cv2.circle(image_bgr, center, 3, (0, 0, 255), -1)

    # Convert BGR to RGB for displaying with matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Display the image with centers
    ax.imshow(image_rgb)
    ax.set_title("Identified Points with OpenCV")


def plot_voronoi(points: np.ndarray[np.float64], ax: plt.Axes) -> None:
    # Create a Voronoi diagram
    vor = Voronoi(points)
    ax.set_title("Voronoi Diagram")
    voronoi_plot_2d(vor, ax=ax)
    ax.invert_yaxis()


def main():
    fig, axs = setup_plot()
    points = rand_points()
    img = get_scatter_plot(points, axs[0])
    centers, image_cv = extract_points(img)
    plot_centers(centers, axs[1], image_cv)
    plot_voronoi(centers, axs[2])

    # Turn on axis
    for ax in axs:
        ax.axis("on")
    fig.savefig(f"images/voronoi{len(points)}.png")

    plt.show()


if __name__ == "__main__":
    main()
