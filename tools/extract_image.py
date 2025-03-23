import os
import cv2
import random
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde



class ImageDataset:
    """
    A class to handle image datasets for visualization and analysis.

    This class provides methods to load images, extract pixel intensity distributions,
    visualize images alongside their intensity histograms, and randomly select images from a folder.

    Instance Attributes
    -------------------
    ```
    - image_path_list : list[str]
        List of paths to images in the specified directory.
    ```

    Methods
    -------
    ```
    - __getitem__(key: int) -> str
    - __len__() -> int
    - visualize_image(image_path: str, output_directory: str) -> None
    ```

    Static Methods
    --------------
    ```
    - extract_distribution_from_image(image_path: str) -> tuple[np.ndarray, gaussian_kde, np.ndarray]
    - get_random_images(folder_path: str, n: int) -> list[str]
    ```
    """

    def __init__(self, image_directory: str) -> None:
        """
        Initializes ImageDataset with paths of images from a directory.

        Parameters:
        -----------
        image_directory : str
            Path to the directory containing image files.
        """
        self.image_path_list = [
            os.path.join(image_directory, file_name) for file_name in os.listdir(image_directory)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __getitem__(self, key: int) -> str:
        """
        Retrieves the image path at the specified index.

        Parameters:
        -----------
        key : int
            Index of the image path to retrieve.

        Returns:
        --------
        str
            Path to the image file.
        """
        return self.image_path_list[key]

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
        --------
        int
            Number of images.
        """
        return len(self.image_path_list)

    @staticmethod
    def extract_distribution_from_image(image_path: str) -> Tuple[np.ndarray, gaussian_kde, np.ndarray]:
        """
        Extracts pixel intensity distribution from an image.

        Parameters:
        -----------
        image_path : str
            Path to the image file.

        Returns:
        --------
        tuple[np.ndarray, gaussian_kde, np.ndarray]
            Normalized pixel values, KDE function, and original grayscale image.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        flattened_values = image.flatten().astype(np.float32) + 0.01
        normalized_values = flattened_values / 255
        kde_values = gaussian_kde(normalized_values)

        return normalized_values, kde_values, image

    def visualize_image(self, image_path: str, output_directory: str) -> None:
        """
        Visualizes the image alongside its histogram and KDE plot.

        Parameters:
        -----------
        image_path : str
            Path to the image file to visualize.
        output_directory : str
            Directory where the visualization image will be saved.

        Returns:
        --------
        None
        """
        os.makedirs(output_directory, exist_ok=True)
        normalized_values, kde, original_image = self.extract_distribution_from_image(image_path)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].imshow(original_image, cmap='gray')
        ax[0].axis("off")
        ax[0].set_title(f"Original Image: {image_path}")

        x_values = np.linspace(min(normalized_values), max(normalized_values), 1000)
        kde_values = kde(x_values)

        ax[1].hist(normalized_values, bins=50, density=True, alpha=0.3, label="Histogram")
        ax[1].plot(x_values, kde_values, color='red', label="KDE")

        ax[1].set_xlabel("Normalized Pixel Intensity")
        ax[1].set_ylabel("Density")
        ax[1].set_title("Histogram & KDE")
        ax[1].legend()

        plt.tight_layout()
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_directory, f"{base_name}_hist_kde.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved plot: {save_path}")

    @staticmethod
    def get_random_images(folder_path: str, n: int) -> List[str]:
        """
        Selects a specified number of random images from a folder.

        Parameters:
        -----------
        folder_path : str
            Directory to select images from.
        n : int
            Number of random images to select.

        Returns:
        --------
        List[str]
            List of randomly selected image filenames.
        """
        all_images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not all_images:
            print("No images found in the directory.")
            return []

        return random.sample(all_images, min(n, len(all_images)))


if __name__ == '__main__':
    # Declare images directory
    image_directory = 'image_dataset/images'

    # Declare object
    image_dataset = ImageDataset(image_directory = image_directory)

    image_dataset.visualize_image(image_dataset[0], output_directory='output_extract')

    pass
