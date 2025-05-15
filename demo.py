import argparse
import pathlib
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render

CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet' / 'Gaze360' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
     
    cap = cv2.VideoCapture(cam)
    # 解析度設定（例如寬 640, 高 480）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution set to: {int(width)}x{int(height)}")
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)
                continue
            # === 新增：處理 IR 相機格式 ===
            if frame is not None:
                if frame.shape[-1] == 2:
                    # YUYV 格式（YUV422 packed）
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
                elif len(frame.shape) == 2 or frame.shape[2] == 1:
                    # 單通道灰階
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # # Process frame
            # results = gaze_pipeline.step(frame)
            # === 防止偵測失敗導致崩潰 ===
            try:
                results = gaze_pipeline.step(frame)
            except Exception as e:
                print("[Warning] Gaze pipeline failed on this frame:", e)
                continue

            # Visualize output
            frame = render(frame, results)
            # print(frame)
            print("Frame shape:", frame.shape)
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()

    #  python demo.py --snapshot models/L2CSNet_gaze360.pkl --devicce cpu --cam 0
    # 480*640 11 FPS
    # 1080*1920 7 FPS
    # python demo.py --snapshot models/L2CSNet_gaze360.pkl --device mps --cam 0
    # 480*640 20~22 FPS
    # 1080*1920 10 FPS