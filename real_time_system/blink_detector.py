import cv2
import time
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import queue
import threading

class EyeBlinkDetector(threading.Thread):
    def __init__(self, blink_queue, ear_threshold=0.25, long_blink_sec=2.0):
        super().__init__(daemon=True)
        self.blink_queue = blink_queue
        self.mp_face_mesh = mp.solutions.face_mesh
        # Use static_image_mode=False for video and specify lower confidence thresholds
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        # MediaPipe landmarks for eyes
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.EAR_THRESHOLD = ear_threshold
        self.LONG_BLINK_THRESHOLD = long_blink_sec
        self.blinking = False
        self.blink_start_time = None
        self.blink_times = []
    
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def classify_blink_sequence(self):
        now = time.time()
        self.blink_times = [t for t in self.blink_times if now - t < 2.0]

        if len(self.blink_times) == 2:
            return "double"
        elif len(self.blink_times) == 3:
            return "triple"
        else:
            return "single"

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera access failed")
            return

        print("[INFO] Blink detector started")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            # Flip horizontally for selfie view
            frame = cv2.flip(frame, 1)
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # To improve performance, mark image as not writeable
            rgb_frame.flags.writeable = False
            results = self.face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # Display the frame with debug info
            debug_frame = frame.copy()
            cv2.putText(debug_frame, "EyeBlinkDetector Active", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                      
            # Process face landmarks if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ih, iw, _ = frame.shape
                    left_eye = [(int(face_landmarks.landmark[i].x * iw),
                                int(face_landmarks.landmark[i].y * ih)) for i in self.LEFT_EYE]
                    right_eye = [(int(face_landmarks.landmark[i].x * iw),
                                int(face_landmarks.landmark[i].y * ih)) for i in self.RIGHT_EYE]

                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    # Draw eyes on debug frame
                    for coord in left_eye + right_eye:
                        cv2.circle(debug_frame, coord, 2, (0, 255, 0), -1)
                    
                    # Display EAR value
                    cv2.putText(debug_frame, f"EAR: {ear:.2f}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Detect blinking based on EAR
                    if ear < self.EAR_THRESHOLD:
                        if not self.blinking:
                            self.blinking = True
                            self.blink_start_time = time.time()
                    else:
                        if self.blinking:
                            blink_duration = time.time() - self.blink_start_time
                            self.blinking = False
                            if blink_duration >= self.LONG_BLINK_THRESHOLD:
                                self.blink_queue.put("long")
                            else:
                                self.blink_times.append(time.time())
                                blink_type = self.classify_blink_sequence()
                                self.blink_queue.put(blink_type)
                                print(f"[INFO] Detected {blink_type} blink")
            
            # Show the debug frame
            cv2.imshow('Eye Blink Detector', debug_frame)
            
            # Add a more reliable exit condition with lower wait time
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                print("[INFO] Exiting...")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("[INFO] Starting Eye Blink Detector...")
    q = queue.Queue()
    
    try:
        blink_detector = EyeBlinkDetector(q)
        blink_detector.start()
        
        # Main thread processes the detected blinks
        while True:
            try:
                blink_type = q.get(timeout=1.0)
                print(f"Main thread received: {blink_type} blink")
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")