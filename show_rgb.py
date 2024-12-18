import numpy as np
import cv2

def read_rgb_binary(filename, width, height):
    # Read binary file
    with open(filename, 'rb') as f:
        binary_data = f.read()
    
    # Convert to numpy array
    img_array = np.frombuffer(binary_data, dtype=np.uint8)
    
    # Reshape to image dimensions with RGB channels
    img = img_array.reshape((height, width, 3))
    
    # Convert from RGB to BGR for OpenCV
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img.shape)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    
    return img

if __name__ == "__main__":
    # Example usage
    filename = "rga_output.bin"
    width = 320
    height = 480
    try:
        img = read_rgb_binary(filename, width, height)
        cv2.imwrite("output.png", img)
        print(f"Successfully read binary file and saved as output.png")
    except Exception as e:
        print(f"Error: {str(e)}")
