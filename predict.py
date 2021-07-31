from model import resnet50
import torch
from PIL import Image
import os
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型权重下载路径在网盘：
# 链接：https://pan.baidu.com/s/1mlmEgK67RTRjqZXwvFMUrw
# 提取码：7dqt
model_dir = 'resNet50-0.8014583333333333.pth'
model =resnet50(num_classes=11).to(device)
model.load_state_dict(torch.load(model_dir, map_location=device))
model.eval()

# 开始预测图片
imgdir = os.path.join(os.path.dirname(__file__), 'val')
filename_list = os.listdir(imgdir)
for filename in filename_list:
    filepath = os.path.join(imgdir, filename)
    # 图片输入模型格式为(n, 3, 224, 224)
    img = Image.open(filepath)
    img = np.expand_dims(np.array(img), 0)
    img = np.transpose(img, [0, 3, 1, 2]).astype(np.float32)
    img /= 255
    outputs = model(torch.from_numpy(img).to(device))[0]
    outputs = torch.sigmoid(outputs)
    print(outputs)