import cv2
import numpy as np
import collections
import warnings
warnings.filterwarnings('ignore')

print("Initializing enhanced age/gender detection...")
print("✓ Using OpenCV face detection + optimized Caffe models\n")

# Load OpenCV face detector (more reliable than external libs)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load Caffe age/gender models
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

try:
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
    print("✓ All models loaded successfully\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def detect_and_predict(frame):
    """
    Detect faces using Haar Cascade and predict age/gender
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        face_predictions = []
        
        for (x, y, w, h) in faces:
            # Add padding for better feature extraction
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face = frame[y1:y2, x1:x2]
            
            if face.shape[0] > 0 and face.shape[1] > 0:
                # Prepare blob for neural networks
                blob = cv2.dnn.blobFromImage(
                    face, 1.0, (227, 227),
                    MODEL_MEAN_VALUES,
                    swapRB=False
                )
                
                # Predict gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender_idx = gender_preds[0].argmax()
                gender = GENDER_LIST[gender_idx]
                gender_conf = gender_preds[0][gender_idx]
                
                # Predict age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age_idx = age_preds[0].argmax()
                age = AGE_LIST[age_idx]
                age_conf = age_preds[0][age_idx]
                
                face_predictions.append({
                    'bbox': (x1, y1, x2, y2),
                    'age': age,
                    'gender': gender,
                    'age_conf': age_conf,
                    'gender_conf': gender_conf
                })
        
        return face_predictions
    except Exception as e:
        print(f"Detection error: {e}")
        return []


def draw_cool_ui(frame, x1, y1, x2, y2, label, box_color=(0, 255, 255)):
    """Draw bounding box with modern UI elements"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    line_l = int((x2 - x1) * 0.15)
    cv2.line(frame, (x1, y1), (x1 + line_l, y1), box_color, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + line_l), box_color, 3)
    cv2.line(frame, (x2, y1), (x2 - line_l, y1), box_color, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + line_l), box_color, 3)
    cv2.line(frame, (x1, y2), (x1 + line_l, y2), box_color, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - line_l), box_color, 3)
    cv2.line(frame, (x2, y2), (x2 - line_l, y2), box_color, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - line_l), box_color, 3)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1 - 40), (x1 + tw + 20, y1), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, label, (x1 + 10, y1 - 12), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)



def live_webcam():
    print("="*70)
    print("🎯 ENHANCED AGE/GENDER DETECTION - OpenCV + Caffe Models")
    print("="*70)
    print("✓ Robust Haar Cascade face detection")
    print("✓ Fast Caffe neural networks")
    print("✓ No external library downloads")
    print("✓ Real-time processing\n")
    print("Press 'q' or 'Esc' to quit\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    age_history = collections.deque(maxlen=12)
    gender_history = collections.deque(maxlen=12)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Resize for faster processing
        display_frame = cv2.resize(frame, (800, 600))
        frame = cv2.resize(frame, (640, 480))
        
        # Run detection
        face_predictions = detect_and_predict(frame)
        
        if face_predictions:
            for pred in face_predictions:
                x1, y1, x2, y2 = pred['bbox']
                # Scale to display frame
                scale_x = display_frame.shape[1] / frame.shape[1]
                scale_y = display_frame.shape[0] / frame.shape[0]
                x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                
                age = pred['age']
                gender = pred['gender']
                age_conf = pred['age_conf']
                gender_conf = pred['gender_conf']
                
                age_history.append(age)
                gender_history.append(gender)
                
                # Vote for most common prediction (smoothing)
                if len(age_history) >= 4:
                    age_result = max(set(age_history), key=age_history.count)
                    gender_result = max(set(gender_history), key=gender_history.count)
                    confidence = "✓✓✓"
                else:
                    age_result = age
                    gender_result = gender
                    confidence = "●" * len(age_history)
                
                label = f"{gender_result}, {age_result}  {confidence}"
                box_color = (200, 100, 255) if gender_result == 'Female' else (0, 255, 255)
                
                draw_cool_ui(display_frame, x1, y1, x2, y2, label, box_color)
        
        # Display info
        info = f"Frame: {frame_count} | Faces: {len(face_predictions)}"
        cv2.putText(display_frame, info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Age/Gender Detection (Enhanced)', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Detection closed.")

if __name__ == "__main__":
    live_webcam()
