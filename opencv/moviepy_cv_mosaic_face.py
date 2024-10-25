import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def mosaic(img, x, y, w, h, block_size=15):
    """Apply mosaic effect to a specific region of the image."""
    # Get the region of interest
    roi = img[y:y+h, x:x+w]
    
    # Calculate new dimensions
    h_blocks = h // block_size
    w_blocks = w // block_size
    
    if h_blocks == 0 or w_blocks == 0:
        return img
    
    # Resize down and up to create mosaic effect
    temp = cv2.resize(roi, (w_blocks, h_blocks), interpolation=cv2.INTER_LINEAR)
    mosaic_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Place the mosaic region back into the image
    result = img.copy()
    result[y:y+h, x:x+w] = mosaic_roi
    return result

def process_frame(frame):
    """Process a single frame: detect faces and apply mosaic."""
    # Convert the frame to RGB (OpenCV uses BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Apply mosaic to each detected face
    for (x, y, w, h) in faces:
        frame_rgb = mosaic(frame_rgb, x, y, w, h)
    
    # Convert back to RGB for moviepy
    return cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

def process_video(input_path, output_path):
    """Process the entire video file."""
    try:
        # Load the video
        video = VideoFileClip(input_path)
        
        # Process the video
        processed_video = video.fl_image(process_frame)
        
        # Write the result
        processed_video.write_videofile(output_path, audio=True)
        
        # Close the video files
        video.close()
        processed_video.close()
        
        print(f"Video processing completed. Output saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    input_path = "input.mp4"  # Replace with your input video path
    output_path = "output_video.mp4"  # Replace with desired output path
    process_video(input_path, output_path)

# @ prunp moviepy_cv_mosaic_face.py
# Moviepy - Building video output_video.mp4.
# MoviePy - Writing audio in output_videoTEMP_MPY_wvf_snd.mp3
# MoviePy - Done.
# Moviepy - Writing video output_video.mp4
# Moviepy - Done !
# Moviepy - video ready output_video.mp4
# Video processing completed. Output saved to output_video.mp4

