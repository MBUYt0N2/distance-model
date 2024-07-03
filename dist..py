import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load YOLO model
model = YOLO("/home/shusrith/Downloads/yolov8-pytorch-open-image-v8-v1/yolov8s-oiv7.pt")

calibration_params = np.load("calibration_params.npy", allow_pickle=True).item()
k = calibration_params["K"]
r = calibration_params["R"]
tvec = calibration_params["t"]
p = calibration_params["P"]

def object_point_world_position(u, v, w, h, p, k):
    u1 = u
    v1 = v + h / 2
    fx = k[0, 0]
    fy = k[1, 1]
    height = 1
    angle_a = 0
    angle_b = math.atan((v1 - height / 2) / fy)
    angle_c = angle_b + angle_a
    depth = (height / np.sin(angle_c)) * math.cos(angle_b)

    k_inv = np.linalg.inv(k)
    R_inv = np.linalg.inv(r)  # Invert the rotation matrix
    tvec_new = -R_inv @ tvec  # Compute the new translation vector
    p_inv = np.hstack((R_inv, tvec_new.reshape(-1, 1))) 
    point_c = np.array([u1, v1, 1])
    c_position = np.matmul(k_inv, depth * point_c)
    c_position = np.append(c_position, 1)
    c_position = np.matmul(p_inv, c_position)
    d1 = np.array((c_position[0], c_position[1]), dtype=float)
    return d1


def distance_func(kuang, xw=5, yw=0.1):
    if len(kuang):
        u, v, w, h = kuang[1], kuang[2], kuang[3], kuang[4]
        d1 = object_point_world_position(u, v, w, h, p, k)
    distance = 0
    if d1[0] <= 0:
        d1[:] = 0
    else:
        distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
    return distance, d1

cap = cv2.VideoCapture(0)
l = []
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 15 != 0:
        continue
    results = model(frame)
    # l.append(results)
    detections = results[0].boxes.xyxy  # Extract detections

    for i in range(detections.shape[0]):
        row = detections[i]
        u, v, w, h = (
            int(row[0]),
            int(row[1]),
            int(row[2] - row[0]),
            int(row[3] - row[1]),
        )
        kuang = [0, u, v, w, h]
        distance, _ = distance_func(kuang, p, k)

        # Annotate frame with distance
        cv2.rectangle(frame, (u, v), (u + w, v + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{distance:.2f}m",
            (u, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    # Display the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


# print(l[0][0].boxes.xyxy[0]) 