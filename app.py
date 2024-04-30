import cv2
import numpy as np
import os

def compare_images(reference_image_path, directory_path):
    # Load the reference image
    reference_image = cv2.imread(reference_image_path)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors in the reference image
    kp_ref, des_ref = orb.detectAndCompute(reference_gray, None)
    
    # Create a BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Iterate over the images in the directory
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        
        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors in the image
        kp_img, des_img = orb.detectAndCompute(gray, None)
        
        # Match descriptors
        matches = bf.match(des_ref, des_img)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Calculate a match score
        match_score = len(matches)
        threshold = 200  # Adjust as needed
        
        if match_score > threshold:
            print(f"Image '{filename}' is similar to the reference image.")
        else:
            print(f"Image '{filename}' is not similar to the reference image.")

# Example usage
reference_image_path = "img.jpg"
directory_path = "dir"
compare_images(reference_image_path, directory_path)