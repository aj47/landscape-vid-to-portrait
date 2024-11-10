import cv2
from mtcnn import MTCNN
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
from tqdm import tqdm

def select_video_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    return file_path

def detect_faces_in_video(video_path, skip_frames=2, scale_factor=0.5):
    if not video_path:
        return
    # Initialize MTCNN detector
    detector = MTCNN()

    # Start video capture from file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Define output video dimensions (9:16 aspect ratio)
    output_width = 1080
    output_height = 1920

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create VideoWriter object for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
    out = cv2.VideoWriter('temp.mp4', fourcc, 20.0, (output_width, output_height))

    # Create a list to store processed frames
    processed_frames = []

    # Process each frame with a progress bar
    frame_count = 0
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_count % skip_frames == 0:
            # Downscale for detection
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Perform face detection
            detections = detector.detect_faces(rgb_frame)
            
            # Scale coordinates back up
            for detection in detections:
                box = detection['box']
                detection['box'] = [int(coord / scale_factor) for coord in box]
        else:
            # Use previous detections for skipped frames
            detections = []

        frame_count += 1

        # Find the largest face
        largest_face = None
        largest_area = 0
        for detection in detections:
            x, y, w, h = detection['box']
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = detection

        # Create a blank vertical frame
        vertical_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        if largest_face:
            x, y, w, h = largest_face['box']

            # Crop the largest face
            face_crop = frame[y:y + h, x:x + w]

            # Resize face_crop to fit the bottom part of the vertical frame
            face_crop_resized = cv2.resize(face_crop, (output_width, output_height // 2))

            # Place the resized face crop in the bottom half
            vertical_frame[output_height // 2:, :] = face_crop_resized

            # Crop the top part of the original frame (excluding the face)
            top_crop = frame[:y, :]

            # Resize top_crop to fit the top part of the vertical frame
            top_crop_resized = cv2.resize(top_crop, (output_width, output_height // 2))

            # Place the resized top crop in the top half
            vertical_frame[:output_height // 2, :] = top_crop_resized

        # If no face is detected, just resize the original frame to fit vertically
        else:
            vertical_frame = cv2.resize(frame, (output_width, output_height))

        # Write the vertical frame to the output video
        out.write(vertical_frame)
        processed_frames.append(vertical_frame)

    # Release the capture and temporary video file
    cap.release()
    out.release()

    # Ask the user for the save location
    save_path = filedialog.asksaveasfilename(
        title="Save Vertical Video",
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )

    if save_path:
        # Create VideoWriter object for output with the chosen save path
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # More efficient codec
        out = cv2.VideoWriter(save_path, fourcc, 20.0, (output_width, output_height))

        # Write processed frames to the output video
        for frame in tqdm(processed_frames, desc="Rendering video"):
            out.write(frame)

        # Release the output video
        out.release()

        print(f"Vertical video saved to: {save_path}")
    else:
        print("No save location selected.")

if __name__ == "__main__":
    video_path = select_video_file()
    if video_path:
        detect_faces_in_video(video_path)
    else:
        print("No video file selected.")
