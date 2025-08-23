import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# Define function to calculate similarity between two bounding boxes
def calculate_similarity(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    union = area1 + area2 - intersection
    
    iou = intersection / union if union > 0 else 0
    
    return iou

# Load YOLO model
model = YOLO("yolov8n.pt").to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

# Initialize DeepSORT tracker
def create_tracker():
    return DeepSort(max_age=30)

# List of video files
video_paths = [
    "/Users/sunny/Downloads/VisionQ_Input_1.mp4", 
    "/Users/sunny/Downloads/VisionQ_Input_2.mp4",
    "/Users/sunny/Downloads/VisionQ_Input_3.mp4",
    "/Users/sunny/Downloads/VisionQ_Input_4.mp4"
]

# Camera labels
camera_labels = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]

# Path to the target person's image
target_person_image_path = "/Users/sunny/Downloads/sunanda.png"

# Load the target person image
target_person_img = cv2.imread(target_person_image_path)
target_features = None
global_target_features = None

if target_person_img is None or target_person_img.size == 0:
    print(f"Could not load target person image. Will use first detected person as target.")
    use_first_person = True
else:
    use_first_person = False
    target_results = model(target_person_img)[0]
    for box in target_results.boxes.data:
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and conf > 0.6:
            target_features = box.cpu().numpy()[:4]
            global_target_features = target_features.copy()
            break

    if target_features is None:
        print("No person detected in target image. Using first detected person as target.")
        use_first_person = True

# Set output path for the sequential video
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop_path, "tracked_person_sequential.mp4")

# Parameters
frame_skip = 5
similarity_threshold = 0.3
display_time_per_camera = 5  # Time to display each camera (in seconds)

# Initialize video writers for each camera
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Process each video completely, one after another
for video_idx, video_path in enumerate(video_paths):
    print(f"\n===== Processing video {video_idx+1}: {video_path} =====")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        continue
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create individual output video for this camera
    camera_output_path = os.path.join(desktop_path, f"tracked_person_camera{video_idx+1}.mp4")
    out = cv2.VideoWriter(camera_output_path, fourcc, fps/frame_skip, (width, height))
    
    print(f"Video info: {total_frames} frames at {fps} FPS")
    print(f"Output will be saved to: {camera_output_path}")
    
    # Initialize tracker for this camera
    tracker = create_tracker()
    
    # Process frames
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    print(f"Now displaying and processing Camera {video_idx+1}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        processed_count += 1
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        camera_text = camera_labels[video_idx]
        cv2.putText(frame, camera_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        timestamp = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(frame, timestamp, (frame.shape[1] - 150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        results = model(frame)[0]
        detections = []
        
        target_found = False
        
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            if int(cls) == 0 and conf > 0.5:
                person_box = box.cpu().numpy()[:4]
                
                if use_first_person and not first_person_detected:
                    global_target_features = person_box.copy()
                    first_person_detected = True
                    print(f"Selected first person as target in video {video_idx+1}, frame {frame_count}")
                    print(f"Target features: {global_target_features}")
                    detections.append([[x1, y1, x2, y2], conf, 0])
                    target_found = True
                elif global_target_features is not None:
                    similarity = calculate_similarity(person_box, global_target_features)
                    if similarity > similarity_threshold:
                        detections.append([[x1, y1, x2, y2], conf, 0])
                        target_found = True
                        if processed_count % 50 == 0:
                            print(f"Target found in frame {processed_count} of video {video_idx+1}, similarity: {similarity:.3f}")
        
        tracked_objects = tracker.update_tracks(detections, frame=frame)
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            label = f"Target Person"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Write the frame to the output video
        out.write(frame)
        
        # Show single camera feed
        window_name = f"Processing Camera {video_idx+1} - {camera_labels[video_idx]}"
        cv2.imshow(window_name, frame)
        
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} frames of video {video_idx+1}")
        
        # Exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Clean up resources for this camera
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Finished processing Camera {video_idx+1}: {processed_count} frames processed")
    
    # Exit if 'q' was pressed
    if key == ord('q'):
        break

print("\nAll cameras processed. Individual camera videos saved to desktop.")