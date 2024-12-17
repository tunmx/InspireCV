import cv2
import os
import numpy as np

def save_image_as_bitmap(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert BGR to RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(img.shape)
    
    # Get dimensions
    height, width = img.shape[:2]
    
    # Create output filename with dimensions
    output_filename = f"in0w{width}-h{height}-rgb888.bin"
    
    # Save as binary file
    img.tofile(output_filename)
    
    print(f"Saved bitmap to {output_filename}")
    cv2.imwrite("input.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(img.shape)
    return output_filename

if __name__ == "__main__":
    # Example usage
    image_path = "/Users/tunm/Downloads/data960x640.jpg"  # Replace with your image path
    try:
        output_file = save_image_as_bitmap(image_path)
    except Exception as e:
        print(f"Error: {str(e)}")
