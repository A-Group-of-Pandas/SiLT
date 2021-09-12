import cv2
import torch

from data_processing import normalize, read_data, resize
from sign_recog_cnn import SignRecogCNN

imgs, labels = read_data("images", "labels")
img = imgs[0]
model = SignRecogCNN()
model.load_state_dict(torch.load("sign_recogn_cnn-4", map_location=torch.device("cpu")))
model.eval()
cv2.imshow("img", img)
img = normalize(img[None, ...])
preds = model(torch.tensor(img))
print(preds)
cv2.waitKey(0)
