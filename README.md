# Candy Counting Algorithm Documentation

## Project Overview

This project involves developing an algorithm to count correct candies placed on a rotating plate. The plate rotates in a right-hand direction, and the goal is to ensure accurate counting of candies that are not broken, fully intact, lying flat, and single (not overlapping). The candies are counted specifically in rows 2-5 (counting from the left). The images used for counting may be blurred due to the motion of the plate. The performance of the algorithm is evaluated based on the number of correctly detected candies, with a score penalty for each three misidentified or missed slots.

### Objectives:
- **Count Correct Candies**: Detect and count full, flat, single candies on the plate.
- **Handle Blurred Images**: Account for motion blur caused by the rotating plate.
- **Score Evaluation**: Penalize based on the number of incorrectly identified or empty slots.

## Project Structure

The project consists of the following main components:

- **CandyCounter.py**: Implements the core logic to count correct candies based on detected contours.
- **DataLoader.py**: Handles loading of candy data from a CSV file, which contains the expected number of correct candies.
- **ImageProcessor.py**: Provides methods for loading, preprocessing, and processing images to detect candy contours.
- **Visualizer.py**: Contains functions to visualize the results, including highlighting correct and incorrect candies on the images.
- **main.py**: The main script that orchestrates the loading of data, processing of images, counting of candies, and output of results.

## Requirements

- Python 3.x
- OpenCV
- Pandas
- Matplotlib
- NumPy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/candy-counter.git
2. Install the required dependencies:
   ```bash
   pip install opencv-python pandas matplotlib numpy
## Running the Project
1. Load and Process Images: The images are loaded, preprocessed, and analyzed to detect and count candies.
   ```python
   python main.py
2. Visualize Results: Use the `Visualizer.py` script to see the results of the candy counting, with correct and incorrect candies highlighted.
   ```python
   python Visualizer.py
## Detailed Components
### CandyCounter.py

The `CandyCounter` module is responsible for counting the correct candies detected on the rotating plate. The key functionality involves analyzing the contours detected in the images to ensure that each candy is counted only once, avoiding double-counting.

#### Key Features:

- **Counting Logic**: The module ensures that each detected candy is unique by checking the center of the contour. If the center has already been counted, it is not counted again.
- **Contour Moments**: The module uses contour moments to calculate the center (centroid) of each contour. This centroid is then used to track which candies have already been counted.

#### Methods:

- `__init__()`: Initializes the `CandyCounter` class with an empty set to track counted candies.
- `count_candies(contours)`: Iterates through the contours, counts the correct candies, and prevents double-counting by checking if the candy has already been counted based on its centroid.
  ```python
    import cv2
    
    class CandyCounter:
        def __init__(self):
            self.counted_candies = set()
    
        def count_candies(self, contours):
            correct_candies = 0
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    if (cx, cy) not in self.counted_candies:
                        self.counted_candies.add((cx, cy))
                        correct_candies += 1
            return correct_candies

This code snippet provides the complete implementation of the CandyCounter module, which is designed to accurately count candies while avoiding duplicate counts. The use of contour moments to identify unique candies is crucial for ensuring accuracy in the counting process.

### DataLoader.py

The `DataLoader` module is responsible for loading the candy count data from a CSV file. This data includes the expected number of correct candies for each image, which is essential for validating the performance of the candy detection algorithm.

#### Key Features:

- **CSV Loading**: The module reads a CSV file containing image indices and the corresponding expected candy counts.
- **Data Structure**: The data is loaded into a pandas DataFrame, which allows for easy manipulation and access during processing.

#### Methods:

- `__init__(csv_path)`: Initializes the `DataLoader` class with the path to the CSV file.
- `load_data()`: Reads the CSV file and returns a pandas DataFrame containing the image indices and correct candy counts.
    ```python
    import pandas as pd
    
    class DataLoader:
        def __init__(self, csv_path):
            self.csv_path = csv_path
    
        def load_data(self):
            return pd.read_csv(self.csv_path, header=None, names=['index', 'correct_count'])
This code snippet provides the complete implementation of the CandyCounter module, which is designed to accurately count candies while avoiding duplicate counts. The use of contour moments to identify unique candies is crucial for ensuring accuracy in the counting process.
### ImageProcessor.py

The `ImageProcessor` module is responsible for handling the image processing tasks required to detect candies on the rotating plate. This includes loading images, preprocessing them to reduce noise, detecting candy contours, and filtering these contours to identify valid candies.

#### Key Features:

- **Image Loading**: Loads images from the specified file path using OpenCV.
- **Preprocessing**: Converts images to grayscale and applies Gaussian blur to reduce noise, making it easier to detect candies.
- **Contour Detection**: Uses adaptive thresholding and morphological operations to detect contours in the preprocessed images.
- **Contour Filtering**: Filters the detected contours based on area and aspect ratio to identify valid candies and discard noise or irrelevant shapes.

#### Methods:

- `__init__(image_path)`: Initializes the `ImageProcessor` class with the path to the image file.
- `load_image()`: Loads the image from the specified path in color mode.
- `preprocess_image(image)`: Converts the image to grayscale and applies Gaussian blur for noise reduction.
- `detect_candies(image)`: Applies adaptive thresholding and morphological operations to detect contours representing candies.
- `filter_candies(contours)`: Filters the contours based on size and shape criteria to ensure only valid candies are considered.
    ```python
        import cv2
        import numpy as np
        
        class ImageProcessor:
            def __init__(self, image_path):
                self.image_path = image_path
        
            def load_image(self):
                return cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        
            def preprocess_image(self, image):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (9, 9), 0)
                return blurred
        
            def detect_candies(self, image):
                thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                kernel = np.ones((5, 5), np.uint8)
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
                contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                return contours
        
            def filter_candies(self, contours):
                filtered_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if 61 < area < 900 and 0.7 < aspect_ratio < 1.3:
                        filtered_contours.append(contour)
                return filtered_contours

This code snippet provides the implementation of the ImageProcessor module, which handles the entire image processing pipeline from loading the image to detecting and filtering candy contours. This module is crucial for accurately identifying candies on the rotating plate.

### Conclusion

This project successfully implements an algorithm to count correct candies on a rotating plate, even with challenges like motion blur. The modular structure of the code allows for easy updates and modifications, making it adaptable for different types of candies or similar detection tasks. Each component is designed to handle specific parts of the process, from image loading and processing to contour detection and counting, ensuring the system's flexibility and robustness in various scenarios.


