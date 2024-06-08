import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()

output_layers_names = net.getUnconnectedOutLayersNames()
if len(output_layers_names) == 0:
    print("Error: No unconnected output layers found.")
    exit()

def detect_objects(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers_names)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

    return img, boxes

def calculate_accuracy_percentage(clicked_position, relative_position, box_position, frame_shape):
    clicked_x, clicked_y = clicked_position
    relative_x, relative_y = relative_position
    box_x, box_y = box_position
    
    tattoo_x = box_x + relative_x
    tattoo_y = box_y + relative_y
    
    distance = math.sqrt((clicked_x - tattoo_x) ** 2 + (clicked_y - tattoo_y) ** 2)
    
    max_distance = math.sqrt(frame_shape[0] ** 2 + frame_shape[1] ** 2)
    
    accuracy_percentage = (1 - (distance / max_distance)) * 100
    
    return accuracy_percentage

def select_image():
    root = tk.Tk()
    root.withdraw() 

    file_path = filedialog.askopenfilename() 
    return file_path

cap = cv2.VideoCapture(0)  

tattoo_path = select_image()
if tattoo_path == "":
    print("No image selected. Exiting...")
    exit()

tattoo = cv2.imread(tattoo_path)

location = None
relative_position = None

def select_location(event, x, y, flags, params):
    global location, relative_position, boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        location = (x, y)
        print(f"Clicked at: ({x}, {y})")
        if len(boxes) > 0:
            box_x, box_y, box_w, box_h = boxes[0]
            relative_position = (x - box_x, y - box_y)
            print(f"Relative position: {relative_position}")

while True:
    ret, frame = cap.read()

    frame_with_objects, boxes = detect_objects(frame)

    if location is None:
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', select_location)
        while location is None:
            cv2.imshow('frame', frame_with_objects)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyWindow('frame')

    if relative_position is not None:
        if len(boxes) > 0:
            box_x, box_y, box_w, box_h = boxes[0]
            relative_x, relative_y = relative_position
            tattoo_pos_x = box_x + relative_x - (tattoo.shape[1] // 2)
            tattoo_pos_y = box_y + relative_y - (tattoo.shape[0] // 2)

            top_left_x = min(max(tattoo_pos_x, 0), frame.shape[1] - tattoo.shape[1])
            top_left_y = min(max(tattoo_pos_y, 0), frame.shape[0] - tattoo.shape[0])

            roi = frame[top_left_y:top_left_y + tattoo.shape[0], top_left_x:top_left_x + tattoo.shape[1]]

            tattoo_gray = cv2.cvtColor(tattoo, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(tattoo_gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            tattoo_fg = cv2.bitwise_and(tattoo, tattoo, mask=mask)

            dst = cv2.add(img_bg, tattoo_fg)
            frame_with_tattoo = frame.copy()  
            frame_with_tattoo[top_left_y:top_left_y + tattoo.shape[0], top_left_x:top_left_x + tattoo.shape[1]] = dst

            accuracy_percentage = calculate_accuracy_percentage(location, relative_position, (box_x, box_y), frame.shape)
            print(f"Accuracy percentage: {accuracy_percentage:.2f}%")
        else:
            frame_with_tattoo = frame.copy()  

        cv2.imshow('frame_with_tattoo', frame_with_tattoo)

    cv2.imshow('normal_frame', frame) 

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
