import cv2
import numpy as np

def detect_faces(image_path):
    """
    Detect faces in an image using OpenCV's Haar Cascade classifier.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        tuple: (image with face rectangles drawn, list of face coordinates)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around detected faces
    image_with_faces = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image_with_faces,
            (x, y),
            (x+w, y+h),
            color=(0, 255, 0),
            thickness=2
        )
        
        # Add a label above the rectangle
        cv2.putText(
            image_with_faces,
            f'Face',
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    
    return image_with_faces, faces

def main():
    """
    Main function to demonstrate face detection.
    """
    try:
        # Replace with your image path
        image_path = "path_to_your_image.png"
        
        # Detect faces
        result_image, faces = detect_faces(image_path)
        
        # Print number of faces detected
        print(f"Found {len(faces)} faces!")
        
        # Display the result
        cv2.imshow('Detected Faces', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Optionally save the result
        cv2.imwrite('detected_faces.jpg', result_image)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

## prunp face_detection_cv.py
## Found 3 faces!

