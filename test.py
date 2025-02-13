import torch
import torchvision
import cv2

print(torch.__version__)  # ควรเป็น 2.6.0+cu118
print(torchvision.__version__)  # ควรเป็น 0.21.0+cu118
print(torch.cuda.is_available())  # ควรเป็น True


cv2.imshow("Image", cv2.imread("001.jpg"))
cv2.waitKey(0)
cv2.destroyAllWindows()