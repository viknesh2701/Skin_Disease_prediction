import os
import cv2
import numpy as np
import csv

# Function to extract color histograms from an image
def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    hist = []
    for i in range(3):
        channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist.extend(channel_hist)
    hist = np.array(hist).flatten()
    hist /= np.sum(hist)
    return hist

# Path to the directory containing images
image_directory = 'C:/Users/viknesh/OneDrive/Desktop/New folder/feature'  # Replace with the path to your image directory
output_csv_file = 'color_histograms.csv'  # Replace with the desired output CSV file path
image_files = os.listdir(image_directory)

# Extract color histograms from each image and store in a CSV file
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Histogram'])  # Write header
    for file in image_files:
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            image_path = os.path.join(image_directory, file)
            color_hist = extract_color_histogram(image_path)
            writer.writerow([file, color_hist])

print(f"Color histograms are stored in {output_csv_file}.")
