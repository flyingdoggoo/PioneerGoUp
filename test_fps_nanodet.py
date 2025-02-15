import cv2
import time
from nanodet_clone.nanodet.util import overlay_bbox_cv
from nanodet_clone.demo.demo import Predictor
from nanodet_clone.nanodet.util import cfg, load_config, Logger # đảm bảo bạn import đầy đủ các hàm cần thiết
import torch

# Giả sử bạn đã có file config, model checkpoint và thiết lập device
cfg_path = "nanodet_clone/config/nanodet-plus-m_320.yml"
model_path = "model_/nanodet_model_best_m320_300.pth"
device = torch.device('cpu')

# Load config và tạo predictor
load_config(cfg, cfg_path)
logger = Logger(-1, use_tensorboard=False)
predictor = Predictor(cfg, model_path, logger, device=device)

# Mở video (đường dẫn tới file video hoặc sử dụng camera bằng chỉ số 0)
video_path = "video_demo.mp4"  # hoặc sử dụng 0 cho webcam
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

    meta, res = predictor.inference(frame)
    result_frame = overlay_bbox_cv(meta['raw_img'][0], res[0], cfg.class_names, score_thresh=0.35)

    # out.write(result_frame)
    cv2.imshow('window', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

fps = total_frame / (end_time - start_time)
print('\n')
print("fps: ",fps)

out.release()
cap.release()