import cv2
import time
from ultralytics import YOLO

model = YOLO("models/crack_yolov11n.pt")
model.to('cpu')
video_path = "video_demo.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print("Không thể mở video:", video_path)
    exit()

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()  # annotated_frame là ảnh kiểu numpy
    
    # out.write(annotated_frame)
    cv2.imshow('window', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
fps = total_frame / (end_time - start_time)
print("fps: ",fps)
out.release()
cap.release()