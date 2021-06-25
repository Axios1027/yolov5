import cv2
import torch
from PIL import Image

model = torch.hub.load('yolov5', 'custom',
                       path='myself_weights/0428_5l_10epochs_640.pt', source='local')
model.classes = [0]
model.conf = 0.4  # confidence threshold (0-1)

result = model("shutterstock_586277846.jpeg", size=1280)
print(len(result.pandas().xyxy[0]))
result.show()

r"""
'''获取视频信息'''
cap = cv2.VideoCapture(r'C:\Users\Axios\Downloads\Crowd flow in subway station.mp4')  # 加载视频
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
print(fps, width, height)

'''视频转图片'''
isOpened = cap.isOpened()
i = 0
while(isOpened):
    i = i+1
    flag, frame = cap.read()
    fileName = '%03d' % i+".jpg"
    if i < 510 :continue
    if flag == True:
        # cv2.imwrite(fileName, frame)  # 命名 图片 图片质量，此处文件名必须以图片格式结尾命名
        result = model(frame,size=1280)
        result.render()
        # result.show()
        print(result.pandas().xyxy[0].to_json(orient="records"))
        cv2.imshow("image",result.imgs[0])
        cv2.waitKey(1)
    else:
        break
cap.release()
print('end')
"""
