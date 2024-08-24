import cv2
import numpy as np
import time
import os
import sqlite3
from ultralytics import YOLO
from tensorflow.lite.python.interpreter import Interpreter
import easyocr
import re
from datetime import datetime, timedelta
import tkinter as tk

# Load the YOLO model for vehicle detection
yolo_model = YOLO("yolov8n.pt")

# Load the TFLite model for license plate detection
lp_modelpath = r'C:\Users\shiva\Downloads\detect.tflite'
lp_lblpath = r'C:\Users\shiva\Downloads\labelmap.txt'
min_conf = 0.5

# Load the labels for license plate detection
with open(lp_lblpath, 'r') as f:
    lp_labels = [line.strip() for line in f.readlines()]

# Load the TFLite model and allocate tensors
lp_interpreter = Interpreter(model_path=lp_modelpath)
lp_interpreter.allocate_tensors()

# Get input and output tensors for license plate detection
lp_input_details = lp_interpreter.get_input_details()
lp_output_details = lp_interpreter.get_output_details()

# Get model details for license plate detection
lp_height = lp_input_details[0]['shape'][1]
lp_width = lp_input_details[0]['shape'][2]
lp_float_input = (lp_input_details[0]['dtype'] == np.float32)
lp_input_mean = 127.5
lp_input_std = 127.5

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Open the video file
video_path = r"C:\Users\shiva\Downloads\Untitled video - Made with Clipchamp (3).mp4"
video = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not video.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Check if we got valid video properties
if width <= 0 or height <= 0 or fps <= 0:
    print(f"Error: Invalid video properties. Width: {width}, Height: {height}, FPS: {fps}")
    exit()

# Set the output database file path
db_path = 'vehicle_data.db'

# List of Indian state codes for license plates
INDIAN_STATE_CODES = [
    'AN', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP',
    'WB'
]


def perform_ocr(image):
    results = reader.readtext(image)
    if results:
        return results[0][1]  # Return the text from the first result
    return ""

def is_valid_license_plate(plate):
    plate = re.sub(r'\s+', '', plate)
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}$'
    if re.match(pattern, plate):
        state_code = plate[:2]
        return state_code in INDIAN_STATE_CODES
    return False

def detect_license_plate(vehicle_image):
    image_rgb = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (lp_width, lp_height))
    input_data = np.expand_dims(image_resized, axis=0)

    if lp_float_input:
        input_data = (np.float32(input_data) - lp_input_mean) / lp_input_std

    lp_interpreter.set_tensor(lp_input_details[0]['index'], input_data)
    lp_interpreter.invoke()

    boxes = lp_interpreter.get_tensor(lp_output_details[1]['index'])[0]
    scores = lp_interpreter.get_tensor(lp_output_details[0]['index'])[0]

    best_score = 0
    best_box = None
    for i in range(len(scores)):
        if scores[i] > min_conf and scores[i] > best_score:
            best_score = scores[i]
            best_box = boxes[i]

    if best_box is not None:
        ymin, xmin, ymax, xmax = best_box
        ymin = int(max(1, (ymin * vehicle_image.shape[0])))
        xmin = int(max(1, (xmin * vehicle_image.shape[1])))
        ymax = int(min(vehicle_image.shape[0], (ymax * vehicle_image.shape[0])))
        xmax = int(min(vehicle_image.shape[1], (xmax * vehicle_image.shape[1])))

        license_plate_img = vehicle_image[ymin:ymax, xmin:xmax]
        ocr_result = perform_ocr(license_plate_img)

        return ocr_result, (xmin, ymin, xmax, ymax)

    return None, None

# Initialize SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS vehicle_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    license_plate TEXT,
    vehicle_type TEXT,
    intime TIMESTAMP,
    outtime TIMESTAMP
)
''')
conn.commit()

# Load existing data from database
vehicle_data = {}
cursor.execute("SELECT * FROM vehicle_data")
for row in cursor.fetchall():
    license_plate = row[1]
    vehicle_data[license_plate] = {
        'id': row[0],
        'type': row[2],
        'intime': datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S"),
        'outtime': datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S") if row[4] else None
    }

# Get screen size using tkinter
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Calculate 1/4th of the screen size
display_width = screen_width // 2
display_height = screen_height // 2

# Create a named window
cv2.namedWindow('Vehicle and License Plate Detection', cv2.WINDOW_NORMAL)

# Set the window size
cv2.resizeWindow('Vehicle and License Plate Detection', display_width, display_height)

frame_count = 0
while video.isOpened():
    start_time = time.time()

    ret, frame = video.read()
    if not ret:
        break

    # Vehicle detection using YOLO
    yolo_results = yolo_model(frame, classes=[2,7])  # 2 for 'car', 7 for 'truck'

    # Process YOLO results
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            vehicle_type = yolo_model.names[cls]

            # Draw bounding box and label for vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, vehicle_type, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Extract vehicle image
            vehicle_image = frame[y1:y2, x1:x2]

            # Detect license plate within the vehicle area
            ocr_result, lp_box = detect_license_plate(vehicle_image)

            if ocr_result and lp_box:
                lp_x1, lp_y1, lp_x2, lp_y2 = lp_box
                # Adjust license plate coordinates to frame coordinates
                lp_x1, lp_y1 = x1 + lp_x1, y1 + lp_y1
                lp_x2, lp_y2 = x1 + lp_x2, y1 + lp_y2

                # Draw bounding box for license plate
                cv2.rectangle(frame, (lp_x1, lp_y1), (lp_x2, lp_y2), (255, 0, 0), 2)

                # Validate the license plate
                if is_valid_license_plate(ocr_result):
                    # Remove spaces from the license plate
                    ocr_result = re.sub(r'\s+', '', ocr_result)
                    # Draw OCR result above the license plate
                    cv2.putText(frame, ocr_result, (lp_x1, lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    current_time = datetime.now()

                    if ocr_result in vehicle_data:
                        last_entry = vehicle_data[ocr_result]

                        if last_entry['outtime'] is None:
                            time_difference = current_time - last_entry['intime']

                            if time_difference > timedelta(minutes=10):
                                # Update outtime if more than 10 minutes have passed
                                vehicle_data[ocr_result]['outtime'] = current_time
                                cursor.execute('''
                                UPDATE vehicle_data 
                                SET outtime = ? 
                                WHERE id = ?
                                ''', (current_time.strftime("%Y-%m-%d %H:%M:%S"), last_entry['id']))
                                conn.commit()
                        else:
                            # Create a new entry if the vehicle is re-entering
                            time_difference = current_time - last_entry['outtime']

                            if time_difference > timedelta(minutes=1):
                                cursor.execute('''
                                INSERT INTO vehicle_data (license_plate, vehicle_type, intime, outtime)
                                VALUES (?, ?, ?, ?)
                                ''', (ocr_result, vehicle_type, current_time.strftime("%Y-%m-%d %H:%M:%S"), None))
                                conn.commit()
                                new_id = cursor.lastrowid
                                vehicle_data[ocr_result] = {
                                    'id': new_id,
                                    'type': vehicle_type,
                                    'intime': current_time,
                                    'outtime': None
                                }
                    else:
                        # New vehicle detected
                        cursor.execute('''
                        INSERT INTO vehicle_data (license_plate, vehicle_type, intime, outtime)
                        VALUES (?, ?, ?, ?)
                        ''', (ocr_result, vehicle_type, current_time.strftime("%Y-%m-%d %H:%M:%S"), None))
                        conn.commit()
                        new_id = cursor.lastrowid
                        vehicle_data[ocr_result] = {
                            'id': new_id,
                            'type': vehicle_type,
                            'intime': current_time,
                            'outtime': None
                        }
                else:
                    # If the license plate is not valid, display it in red
                    cv2.putText(frame, f"Invalid: {ocr_result}", (lp_x1, lp_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)

    # Resize the frame to fit the display window
    resized_frame = cv2.resize(frame, (display_width, display_height))

    # Display the resized frame
    cv2.imshow('Vehicle and License Plate Detection', resized_frame)

    # Calculate the time spent processing the frame
    processing_time = time.time() - start_time

    # Print progress
    frame_count += 1
    print(f"Processed frame {frame_count} in {processing_time:.2f} seconds")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
video.release()
cv2.destroyAllWindows()
conn.close()

print(f"Video processing complete. Processed {frame_count} frames.")
print(f"Vehicle data saved in database: '{db_path}'")
