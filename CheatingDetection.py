import cv2
import numpy as np
import mediapipe as mp
import pyaudio
import time
import ssl
import platform
import sys

# Conditional import based on OS
if platform.system() == 'Darwin':  # macOS
    import AppKit
elif platform.system() == 'Windows':
    import win32gui
    import win32process
    import psutil

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

ssl._create_default_https_context = ssl._create_unverified_context

class ExamMonitoringSystem:
    def __init__(self):
        # Constants
        self.NOISE_THRESHOLD = 5
        self.SUS_THRESHOLD = 2  # 2 seconds
        self.MISSING_THRESHOLD = 5
        self.PHONE_CONFIDENCE = 0.15
        self.END_MESSAGE_DURATION = 5
        self.EXAM_END_COUNTDOWN = 10
        self.TIMER_DURATION = 3600

        # Face detection setup
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )

        # Object detection for phone
        self.model = MobileNetV2(weights='imagenet')

        # Timing variables
        self.current_window_title = None
        self.front_face_start_time = None
        self.is_sus = False
        self.sus_start_time = None
        self.missing_start_time = None
        self.prev_frame = None
        self.end_message_start_time = None
        self.exam_end_countdown_start = None
        self.start_time = time.time()

    def check_application_switch(self):
        try:
            # OS-specific application switch detection
            if platform.system() == 'Darwin':  # macOS
                active_app = AppKit.NSWorkspace.sharedWorkspace().activeApplication()
                current_app_name = active_app['NSApplicationName']
                
                if self.current_window_title is None:
                    self.current_window_title = current_app_name
                elif current_app_name != self.current_window_title:
                    print("Application switch detected. Ending exam.")
                    return False
            
            elif platform.system() == 'Windows':
                current_window = win32gui.GetForegroundWindow()
                _, pid = win32process.GetWindowThreadProcessId(current_window)
                current_app = psutil.Process(pid).name()
                
                if self.current_window_title is None:
                    self.current_window_title = current_app
                elif current_app != self.current_window_title:
                    print("Application switch detected. Ending exam.")
                    return False
            
            return True
        except Exception as e:
            print(f"Error checking application switch: {e}")
            return True

    def check_face_orientation(self, face_landmarks, frame):
        try:
            if face_landmarks is None:
                return True

            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[4]
            h, w, _ = frame.shape
            left_eye_x = int(left_eye.x * w)
            right_eye_x = int(right_eye.x * w)
            nose_x = int(nose_tip.x * w)

            return nose_x < left_eye_x or nose_x > right_eye_x
        except Exception as e:
            print(f"Error in face orientation check: {e}")
            return False

    def draw_face_box(self, face_landmarks, frame, is_suspicious):
        try:
            h, w, _ = frame.shape
            face_x = [int(landmark.x * w) for landmark in face_landmarks.landmark]
            face_y = [int(landmark.y * h) for landmark in face_landmarks.landmark]

            x_min, x_max = min(face_x), max(face_x)
            y_min, y_max = min(face_y), max(face_y)

            # Change box color based on suspicious status
            color = (0, 0, 255) if is_suspicious else (255, 0, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        except Exception as e:
            print(f"Error drawing face box: {e}")

    def detect_phone(self, frame):
        try:
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            predictions = self.model.predict(img, verbose=0)
            results = decode_predictions(predictions, top=5)[0]

            phone_labels = [
                'iPhone', 'iPod', 'mobile_phone', 'cellular_telephone', 'hand-held_computer'
            ]

            for _, label, conf in results:
                if label in phone_labels and conf > self.PHONE_CONFIDENCE:
                    return True
            return False
        except Exception as e:
            print(f"Error in phone detection: {e}")
            return False

    def check_audio_level(self):
        try:
            data = np.frombuffer(
                self.stream.read(1024, exception_on_overflow=False),
                dtype=np.float32
            )
            volume_norm = np.linalg.norm(data) * 10
            return volume_norm > self.NOISE_THRESHOLD
        except Exception as e:
            print(f"Error in audio check: {e}")
            return False

    def detect_environment_change(self, frame):
        try:
            if self.prev_frame is None:
                self.prev_frame = frame
                return False

            diff = cv2.absdiff(self.prev_frame, frame)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            non_zero_count = cv2.countNonZero(thresh)

            self.prev_frame = frame
            return non_zero_count > (frame.shape[0] * frame.shape[1] * 0.1)
        except Exception as e:
            print(f"Error in environment change detection: {e}")
            return False

    def display_timer_and_message(self, frame):
        try:
            elapsed_time = int(time.time() - self.start_time)
            remaining_time = max(0, self.TIMER_DURATION - elapsed_time)
            minutes, seconds = divmod(remaining_time, 60)

            cv2.putText(frame, f"Time Remaining: {minutes:02}:{seconds:02}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "Exam Ongoing", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        except Exception as e:
            print(f"Error displaying timer and message: {e}")

    def process_frame(self, frame):
        try:
            # Check application switch
            if not self.check_application_switch():
                return None, []

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(rgb_frame)

            sus_reasons = []
            sus_face_detected = False

            if face_results.multi_face_landmarks:
                self.missing_start_time = None
                num_faces = len(face_results.multi_face_landmarks)

                for face_landmarks in face_results.multi_face_landmarks:
                    face_not_facing_front = self.check_face_orientation(face_landmarks, frame)
                    
                    if face_not_facing_front:
                        sus_reasons.append("Face Not Facing Front")
                        sus_face_detected = True
                    
                    self.draw_face_box(face_landmarks, frame, sus_face_detected)

            else:
                if not self.missing_start_time:
                    self.missing_start_time = time.time()
                elif time.time() - self.missing_start_time > self.MISSING_THRESHOLD:
                    sus_reasons.append("Person Missing")
                    sus_face_detected = True

            phone_detected = self.detect_phone(frame)
            if phone_detected:
                sus_reasons.append("Phone Detected")
                sus_face_detected = True

            loud_noise = self.check_audio_level()
            if loud_noise:
                sus_reasons.append("Loud Noise Detected")
                sus_face_detected = True

            environment_change = self.detect_environment_change(frame)
            if environment_change:
                sus_reasons.append("Environment Change Detected")
                sus_face_detected = True

            self.is_sus = sus_face_detected

            if self.is_sus:
                if not self.sus_start_time:
                    self.sus_start_time = time.time()

                if time.time() - self.sus_start_time > self.SUS_THRESHOLD:
                    if not self.exam_end_countdown_start:
                        self.exam_end_countdown_start = time.time()

                    countdown_time = int(self.EXAM_END_COUNTDOWN - 
                        (time.time() - self.exam_end_countdown_start))

                    cv2.putText(frame, "Suspicious Activity Detected!", 
                                (frame.shape[1] // 4, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.putText(frame, f"Exam will end in {countdown_time} seconds", 
                                (frame.shape[1] // 4, frame.shape[0] // 2 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if countdown_time <= 0:
                        print("Exam ends. Reasons:", ", ".join(sus_reasons))
                        return None, sus_reasons

                for i, reason in enumerate(sus_reasons):
                    cv2.putText(frame, reason, (10, 50 + i * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            else:
                self.sus_start_time = None
                self.exam_end_countdown_start = None

            self.display_timer_and_message(frame)

            return frame, sus_reasons
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, []

    def cleanup(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.face_mesh.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def show_instructions():
    # Get screen resolution
    screen_width, screen_height = cv2.VideoCapture(0).read()[1].shape[1], cv2.VideoCapture(0).read()[1].shape[0]
    
    # Create a full-screen white image for instructions
    instructions = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 0)
    thickness = 1

    # Title
    cv2.putText(instructions, "EXAM MONITORING SYSTEM", 
                (screen_width // 4, screen_height // 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Rules
    rules = [
        "EXAM RULES AND GUIDELINES",
        "1. Keep face forward and centered in camera",
        "2. No mobile phones or electronic devices allowed",
        "3. No talking or unnecessary noise",
        "4. Do not leave the exam area",
        "5. Maintain a stable environment",
        "6. Press 'Q' to quit exam",
        "7. System will monitor for suspicious activities",
        "",
        "CONSEQUENCES OF VIOLATION:",
        "- Continuous suspicious activity will end your exam",
        "- All suspicious activities are logged",
        "",
        "PRESS ENTER TO START EXAM",
        "PRESS ESC TO EXIT"
    ]

    # Render rules
    for i, rule in enumerate(rules):
        cv2.putText(instructions, rule, 
                    (screen_width // 4, screen_height // 5 + i * 40), 
                    font, font_scale, color, thickness)

    return instructions

def main():
    # Show instructions
    instructions = show_instructions()
    cv2.namedWindow('Exam Instructions', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Exam Instructions', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Exam Instructions', instructions)

    # Wait for user input
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            cv2.destroyAllWindows()
            break
        elif key == 27:  # ESC key
            sys.exit()

    # Start exam monitoring
    detector = ExamMonitoringSystem()
    
    try:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame, sus_reasons = detector.process_frame(frame)
            if frame is None:
                break

            cv2.imshow('Cheat Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 9:  # 'q' or Tab key
                break

    except Exception as e:
        print(f"Error in exam monitoring: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()

if __name__ == "__main__":
    main()