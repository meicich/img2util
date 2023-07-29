import cv2
import torch
from PIL import Image

# モデルの読み込み
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 画像の読み込み
img = Image.open('img/test.png')

# 推論
results = model(img)

# 結果の表示
results.print()  # 各オブジェクトのクラス、座標、信頼度を表示

