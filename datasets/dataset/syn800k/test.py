import torch
import torch.nn as nn
import torch.nn.functional as F


img_path = "/home/wx/data/iocr_training/Syn800k/SynthText/200/zoo_146_109.jpg"
image = cv2.imread(img_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

x = F.grid_sample(x, grid)