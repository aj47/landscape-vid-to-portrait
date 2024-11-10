import cv2
from xailient import Xailient
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

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

def detect_faces_in_video(video_path):
    if not video_path:
        return
    # Initialize Xailient model
    model = Xailient()

    # Start video capture from file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face detection
        detections = model.detect(frame)

        # Draw rectangles around detected faces
        for detection in detections:
            x, y, w, h = detection['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            confidence = detection['confidence']
            cv2.putText(frame, f"{confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = select_video_file()
    if video_path:
        detect_faces_in_video(video_path)
    else:
        print("No video file selected.")
