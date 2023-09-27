import os
import pathlib

import gradio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def image_counterfeit_detection(image, n1=9999, n2=9999, n3=9999, anomalies=0):
    import cv2
    import numpy as np
    from skimage import measure
    
    # # Load the satellite image (you need to specify the path)
    # image_path = 'path_to_satellite_image.jpg'
    # image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create a binary mask
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours (noise)
    min_contour_area = 1000  # Adjust this threshold as needed
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    
    # Draw the filtered contours on the original image
    cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), thickness=2)  # Green color for boundaries
    
    # Create an empty mask for drawing boundaries
    boundary_mask = np.zeros_like(thresh)
    
    # Draw the filtered contours on the mask
    cv2.drawContours(boundary_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
    
    # Find connected components in the mask
    label_image = measure.label(boundary_mask)
    # Iterate through connected components and extract their coordinates
    farm_boundaries = []
    for region in measure.regionprops(label_image):
        min_row, min_col, max_row, max_col = region.bbox
        farm_boundaries.append(((min_col, min_row), (max_col, max_row)))
    
    # You now have a list of farm boundaries as coordinate pairs
    print(farm_boundaries)
    
    return str(farm_boundaries), image


def launch():
    gradio.Interface(fn=image_counterfeit_detection,
                     inputs=[gradio.Image(label="Select Image"),
                             gradio.Slider(label="Yellow", value=1501, minimum=1, maximum=10000),
                             gradio.Slider(label="Green", value=3011, minimum=1, maximum=10000),
                             gradio.Slider(label="Red", value=1261, minimum=1, maximum=10000),
                             gradio.Slider(label="Anomalies[optional]", value=0, minimum=1, maximum=10),
                             ],
                     outputs=["text", "image"]).launch(share=True)


if __name__ == '__main__':
    launch()
