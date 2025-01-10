import cv2
import numpy as np
from picamera2 import Picamera2
import math
import time

# Parámetros globales
real_ball_radius = 0.0427  # Radio real de la pelota de fútbol en metros

# Función para segmentar la pelota por color
def segment_ball_by_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 0, 103])  # Ajustar según el color de la pelota
    upper_color = np.array([105, 80, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

# Función para segmentar la portería
def segment_goal(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 221])
    upper = np.array([179, 124, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Detección de líneas de la portería
def detect_goal_lines(mask, frame):
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180, threshold=70, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde
    return frame

# Detección de área de la portería
def detect_goal_area(mask, frame):
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180, threshold=70, minLineLength=50, maxLineGap=10)
    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -10 < angle < 10:
                horizontal_lines.append((x1, y1, x2, y2))
        if horizontal_lines:
            longest_line = max(horizontal_lines, key=lambda l: abs(l[2] - l[0]))
            x1, y1, x2, y2 = longest_line
            width = abs(x2 - x1)
            height = int(width / 3)
            cv2.rectangle(frame, (x1, y1), (x2, y1 + height), (0, 255, 255), 2)
            cv2.putText(frame, "Porteria", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame

# Detección de la pelota usando HoughCircles
def detect_ball_using_hough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=60, minRadius=40, maxRadius=120)
    if circles is not None:
        circles = np.round(circles[0]).astype("int")
        return circles[0]  # Devolver el primer círculo detectado
    return None

# Función principal para la detección
def detect_ball_in_video():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('detected_ball_video.mp4', fourcc, 30, (640, 480))

    old_gray = None
    old_points = None
    ball_center = None
    ball_radius = None
    last_center = None
    speed_real = 0
    frame_count = 0
    start_time = time.time()

    cv2.namedWindow("Detección de la Pelota", cv2.WINDOW_NORMAL)

    while True:
        frame = picam2.capture_array()
        frame_count += 1

        # Calcular FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 1

        if frame_count % 20 == 0 or old_gray is None:
            mask = segment_ball_by_color(frame)
            best_circle = detect_ball_using_hough(frame)
            if best_circle is not None:
                ball_center = (best_circle[0], best_circle[1])
                ball_radius = best_circle[2]
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old_points = np.array([[ball_center]], dtype=np.float32) if ball_center is not None else None
            last_center = ball_center

        if old_points is not None:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), old_points, None)
            if status[0][0] == 1:
                new_ball_center = tuple(new_points[0].ravel())
                if last_center is not None:
                    dist_px = np.linalg.norm(np.array(new_ball_center) - np.array(last_center))
                    if ball_radius is not None and ball_radius > 0:
                        scale = real_ball_radius / ball_radius
                        speed_real = (dist_px * scale) * fps
                    last_center = new_ball_center
            old_points = new_points
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        goal_mask = segment_goal(frame)
        frame_with_goal = detect_goal_lines(goal_mask, frame)
        frame_with_goal = detect_goal_area(goal_mask, frame_with_goal)

        cv2.putText(frame_with_goal, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_with_goal, f"Velocidad: {speed_real:.2f} m/s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detección de la Pelota", frame_with_goal)
        out.write(frame_with_goal)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    out.release()
    cv2.destroyAllWindows()

# Ejecutar la función
detect_ball_in_video()
