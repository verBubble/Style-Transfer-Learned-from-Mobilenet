import cv2
import numpy as np
import torch
from imutils import paths
from transformer_net import TransformerNet 
from PIL import Image
from torchvision import transforms

model = TransformerNet()
model.load_state_dict(torch.load('checkpoints/GodBearer.pth'))
model.cuda()
model.eval()

trm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

cap = cv2.VideoCapture(0)
while(True):
	success,img = cap.read()
	img = Image.fromarray(img).resize((512,512))
	img = np.array(img)
	cv2.imshow("before",img)

	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img).resize((256,256))

	

	img = trm(img).cuda()
	t_img = model(img.unsqueeze(0)).squeeze(0).cpu()

	t_img /=255
	t_img[t_img > 1] = 1
	t_img[t_img < 0] = 0
	img = transforms.ToPILImage()(t_img)
	img = img.resize((512,512))
	img = np.array(img)

	cv2.imshow("after",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

	k = cv2.waitKey(1)
	if k == 27:
		cv2.destroyAllWindows()
		break

cap.release()