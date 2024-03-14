import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load pre-trained MobileNetV2 model for object detection
model = MobileNetV2(weights='imagenet', include_top=True)

# Function to detect objects in frames using TensorFlow
def detect_objects(frame):
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = preprocess_input(processed_frame)
    processed_frame = np.expand_dims(processed_frame, axis=0)
    predictions = model.predict(processed_frame)
    return predictions

# Function to calculate anomaly score
def calculate_anomaly_score(object_info):
    # Placeholder: Implement anomaly scoring based on object behavior or attributes
    # For demonstration purposes, we'll assume a random score between 0 and 1
    anomaly_score = np.random.random()
    return anomaly_score

# Function to draw bounding boxes and anomaly indicators on frame
def draw_boxes(frame, object_info, anomaly_score):
    # Placeholder: Implement drawing bounding boxes and anomaly indicators
    # For demonstration purposes, we'll just print the anomaly score on the frame
    cv2.putText(frame, f'Anomaly Score: {anomaly_score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

# Function to establish baseline for normal activity using background subtraction
def establish_baseline(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None

    # Use first few frames to establish baseline
    baseline_frame = None
    num_frames = 50
    count = 0
    while cap.isOpened() and count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if baseline_frame is None:
            baseline_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            baseline_frame = cv2.addWeighted(baseline_frame, 0.9, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0.1, 0)
        count += 1

    cap.release()
    return baseline_frame

# Main function for video stream analysis
def analyze_video_stream(video_path, baseline_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate absolute difference between current frame and baseline frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(frame_gray, baseline_frame)

        # Thresholding and morphological operations to get foreground mask
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

        # Detect objects in the frame
        object_info = detect_objects(frame)

        # Calculate anomaly score for detected objects
        anomaly_score = calculate_anomaly_score(object_info)

        # Draw bounding boxes and anomaly indicators on frame
        frame_with_boxes = draw_boxes(frame, object_info, anomaly_score)

        # Display frame with bounding boxes and anomaly indicators
        cv2.imshow('Video Stream', frame_with_boxes)

        # Check for user input to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to trigger alerts based on anomaly thresholds
def trigger_alert(anomaly_score, threshold=0.5):
    if anomaly_score > threshold:
        print("Alert: Anomaly Detected!")

# Run the program
if __name__ == "__main__":
    # Set the path to the input video file
    video_path = "C:\\Users\\vihar\\Dropbox\\My PC (LAPTOP-1FLQ6SIH)\\Pictures\\Camera Roll\\indhu self introduction.mp4"

    # Establish baseline for normal activity using background subtraction
    baseline_frame = establish_baseline(video_path)
    print(baseline_frame)

    # Analyze video stream for anomalies
    print(analyze_video_stream(video_path, baseline_frame))
