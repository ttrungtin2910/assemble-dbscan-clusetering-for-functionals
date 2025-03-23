import cv2
import os
from tqdm import tqdm
import numpy as np

class VideoFrameExtractor:
    """
    A class to extract frames from video files (.MOV) and save cropped images.

    This class provides functionality to extract one frame per second from each .MOV file
    in a specified folder and save cropped square frames into an output folder.

    Instance Attributes
    -------------------
    ```
    - video_folder : str
        Path to the folder containing input .MOV video files.

    - output_folder : str
        Path to the folder where extracted images will be saved.
    ```

    Methods
    -------
    ```
    - extract_frames_from_videos() -> None
    - extract_frames(video_path: str, video_name: str) -> None
    - save_image(image: np.ndarray, video_name: str, image_count: int) -> None
    ```

    Static Methods
    --------------
    ```
    - crop_square_from_bottom(image: np.ndarray) -> np.ndarray
    ```
    """

    def __init__(self, video_folder, output_folder):
        """
        Initializes VideoFrameExtractor with specified video and output folders.

        Parameters:
        -----------
        video_folder : str
            Path to the folder containing input .MOV video files.
        output_folder : str
            Path to the folder where extracted images will be saved.
        """
        self.video_folder = video_folder
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_frames_from_videos(self):
        """
        Processes all .MOV video files in the video folder, extracting frames.

        Returns:
        --------
        None
        """
        video_files = [f for f in os.listdir(self.video_folder) if f.lower().endswith(".mov")]

        if not video_files:
            print("No .MOV files found in the directory.")
            return

        for video_file in video_files:
            video_path = os.path.join(self.video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]

            print(f"\nProcessing video: {video_file}")
            self.extract_frames(video_path, video_name)

    def extract_frames(self, video_path: str, video_name: str) -> None:
        """
        Extracts one frame per second from the given video file.

        Parameters:
        -----------
        video_path : str
            Path to the video file to process.
        video_name : str
            Name of the video file without extension.

        Returns:
        --------
        None
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames // fps

        print(f"Video FPS: {fps}, Estimated Duration: {duration} seconds")

        frame_count = 0
        image_count = 0

        with tqdm(total=duration, desc=f"Extracting {video_name}", unit="s") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % fps == 0:
                    cropped_frame = self.crop_square_from_bottom(frame)
                    self.save_image(cropped_frame, video_name, image_count)
                    image_count += 1
                    pbar.update(1)

                frame_count += 1

        cap.release()
        print(f"Completed processing: {video_name}")

    @staticmethod
    def crop_square_from_bottom(image: np.ndarray) -> np.ndarray:
        """
        Crops a square image from the bottom of the given image frame.

        Parameters:
        -----------
        image : np.ndarray
            Image frame to crop.

        Returns:
        --------
        np.ndarray
            Cropped square image.
        """
        height, width, _ = image.shape
        square_size = min(height, width)

        x_start = (width - square_size) // 2
        y_start = height - square_size

        return image[y_start:y_start + square_size, x_start:x_start + square_size]

    def save_image(self, image: np.ndarray, video_name: str, image_count: int) -> None:
        """
        Saves the cropped image frame to the output folder.

        Parameters:
        -----------
        image : np.ndarray
            Cropped image frame.
        video_name : str
            Name of the video file without extension.
        image_count : int
            Index number of the image frame.

        Returns:
        --------
        None
        """
        image_path = os.path.join(self.output_folder, f"{video_name}_frame_{image_count}.jpg")
        cv2.imwrite(image_path, image)


if __name__ == "__main__":
    video_folder = "videos"
    output_folder = "images"

    extractor = VideoFrameExtractor(video_folder, output_folder)
    extractor.extract_frames_from_videos()
