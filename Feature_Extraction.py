import os
import cv2
import numpy as np

# Class to compute Local Binary Patterns
class LocalBinaryPatterns:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = np.array([], dtype=np.uint8)

        for y in range(self.radius, gray.shape[0] - self.radius):
            for x in range(self.radius, gray.shape[1] - self.radius):
                roi = gray[y - self.radius:y + self.radius + 1, x - self.radius:x + self.radius + 1]
                lbp = np.append(lbp, self.compute_lbp(roi))

        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Normalize the histogram
        return hist

    def compute_lbp(self, roi):
        center = roi[self.radius, self.radius]
        binary = (roi >= center) * 1
        num = np.packbits(binary.flatten())
        return num[0]

# Path to the directory containing images
image_directory = 'C:/Users/viknesh/OneDrive/Desktop/New folder/feature' 
output_csv_file = 'LBP.csv'  # Replace with the path to your image directory
image_files = os.listdir(image_directory)

# Extract LBP features from each image
lbp_features = []
desc = LocalBinaryPatterns(num_points=8, radius=1)
for file in image_files:
    if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
        image_path = os.path.join(image_directory, file)
        image = cv2.imread(image_path)
        lbp_hist = desc.describe(image)
        lbp_features.append(lbp_hist)

# Convert the features to a numpy array
lbp_features = np.array(lbp_features)

# Display the extracted features
print("LBP Features:")
print(lbp_features)
