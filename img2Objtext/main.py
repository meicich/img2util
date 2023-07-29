import cv2
import torch
from PIL import Image

# モデルの読み込み
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 画像の読み込み
img_path = 'img/test2.png'
img = Image.open(img_path)

# 画像の推論
# confとiouの閾値を調整
results = model(img)  # まずは推論を実行

# 検出結果の閾値を調整
predictions = results.pandas().xyxy[0]  # pandas形式で結果を取得
predictions = predictions[predictions['confidence'] > 0.25]  # confidenceが0.25以上の結果だけを取得

# 確信度とクラスラベルを表示
for _, prediction in predictions.iterrows():
    xmin, ymin, xmax, ymax, confidence, class_id, class_name = prediction
    print(f"Class: {class_name}, Confidence: {confidence}, Bounding Box: {(xmin, ymin, xmax, ymax)}")

