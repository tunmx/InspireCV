import numpy as np
import cv2

def read_rgb_binary(filename):
    # Extract dimensions from filename using string parsing
    # Format: out0w480-h320-rgb888.bin
    parts = filename.split('w')[1].split('-h')
    width = int(parts[0])
    height = int(parts[1].split('-')[0])
    
    # Read binary file
    with open(filename, 'rb') as f:
        binary_data = f.read()
    
    # Convert to numpy array
    img_array = np.frombuffer(binary_data, dtype=np.uint8)
    
    # Reshape to image dimensions with RGB channels
    img = img_array.reshape((height, width, 3))
    
    # Convert from RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img_bgr.shape)

    cv2.imshow("img", img_bgr)
    cv2.waitKey(0)
    
    return img_bgr

if __name__ == "__main__":
    # Example usage
    filename = "out0w320-h480-rgb888.bin"
    try:
        img = read_rgb_binary(filename)
        cv2.imwrite("output.png", img)
        print(f"Successfully read binary file and saved as output.png")
    except Exception as e:
        print(f"Error: {str(e)}")
