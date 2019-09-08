import cv2
import numpy as np
import torch
import time
from PIL import Image
import torchvision.transforms as transforms

# Neural_Style
from transformer_net import TransformerNet 

# define Neural_style_transfer model and transform tool 
def NeuralStyle_init(weight_path, alpha):
    model = TransformerNet(alpha)
    model.load_state_dict(torch.load(weight_path))
    model.cuda()
    model.eval()
    return model 

# define how to transform a image
# 'img' here is a cv2.VideoCapture return
def transform(img, model):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)).astype(np.float32)
    img = torch.from_numpy(img.transpose(2,0,1))
    img = img.cuda()

    # style transfer
    t_img = model(img.unsqueeze(0)).data.squeeze(0).cpu()

    # process after transferring
    #t_img /=255  
    t_img[t_img > 255] = 255
    t_img[t_img < 0] = 0
    img = t_img.numpy().transpose(1,2,0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# This is the main function
# open camera
cap = cv2.VideoCapture('Video/input.avi')  
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frm_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out_trm = cv2.VideoWriter('Video/outpy_trm.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (256,256))

# define model and trm here
model = NeuralStyle_init("checkpoints/201/0_style.pth", 0.1)
cnt = 0
while(True): 
    ret, frame = cap.read()
    cnt += 1
    if ret == True:
        start = time.time()
        # style transfor
        frame_trm = transform(frame, model)
        # write the frame after style transfer into file
        out_trm.write(frame_trm)
        end = time.time()
        print('[{}/{}], fps: {}'.format(cnt, frm_cnt, 1/(end-start)))
    else: 
        break
# When everything done, release the video capture and video write objects
cap.release()
out_trm.release()
