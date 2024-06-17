# -*- coding: utf-8 -*-


import os
import cv2
import numpy as np
import csv

# Function to extract contour-based features from an image
def extract_contour_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    total_area = sum(contour_areas)
    contour_ratios = [area / total_area for area in contour_areas]
    return contour_areas, contour_ratios

# Path to the directory containing images
image_directory = 'C:/Users/viknesh/OneDrive/Desktop/New folder/feature'  # Replace with the path to your image directory
output_csv_file = 'contour_features.csv'  # Replace with the desired output CSV file path
image_files = os.listdir(image_directory)

# Extract contour-based features from each image and store in a CSV file
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'ContourAreas', 'ContourRatios'])  # Write header
    for file in image_files:
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            image_path = os.path.join(image_directory, file)
            contour_areas, contour_ratios = extract_contour_features(image_path)
            writer.writerow([file, contour_areas, contour_ratios])

print(f"Contour-based features are stored in {output_csv_file}.")
