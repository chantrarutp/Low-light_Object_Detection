
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt  # เพิ่มไลบรารีสำหรับแสดงภาพ

# กำหนดอุปกรณ์ (ใช้ GPU ถ้ามี)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โครงสร้างของโมเดล Zero-DCE ที่ตรงกับ Epoch99.pth
class ZeroDCE(nn.Module):
    def __init__(self):
        super(ZeroDCE, self).__init__()

        # โครงสร้างที่ตรงกับการฝึกใน Epoch99.pth
        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1))) * 2.5
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + r6*(torch.pow(x,2)-x)
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return enhance_image_1,enhance_image,r

# โหลดโมเดลจากไฟล์
model_path = "Epoch99.pth"  # ใส่ path ที่คุณบันทึกไฟล์ Epoch99.pth
model = ZeroDCE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ฟังก์ชันปรับปรุงภาพแสงน้อย
def enhance_image(image_path, model, transform, device):
    original_image = Image.open(image_path).convert("RGB")  # โหลดภาพต้นฉบับ
    original_size = original_image.size  # เก็บขนาดต้นฉบับ (width, height)

    image = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_image = model(image)[0].squeeze(0).cpu()  # ใช้ 'enhance_image_1' จากโมเดล

    # แปลงกลับเป็น PIL Image
    enhanced_image = transforms.ToPILImage()(enhanced_image)

    # 🔥 **Resize กลับไปเป็นขนาดเดิม**
    enhanced_image = enhanced_image.resize(original_size, Image.BILINEAR)

    return original_image, enhanced_image  # คืนค่าภาพต้นฉบับ + ภาพที่ปรับปรุงแล้ว

# การแปลงภาพ
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# เลือกไฟล์ภาพจากคอมพิวเตอร์
root = tk.Tk()
root.withdraw()  # ซ่อนหน้าต่างหลัก
file_path = filedialog.askopenfilename(title="เลือกภาพที่ต้องการปรับปรุง", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

# ปรับปรุงภาพ
if file_path:
    original_image, enhanced_image = enhance_image(file_path, model, transform, device)

    # แปลงภาพ PIL เป็น OpenCV
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    enhanced_cv = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

    # ปรับขนาดหน้าต่างให้พอดีกับภาพ แต่ไม่เกิน 800x800 px
    max_size = 700
    h, w = original_cv.shape[:2]
    scale = min(max_size / max(h, w), 1.0)  # คำนวณ scale factor โดยไม่เสียอัตราส่วน
    new_size = (int(w * scale), int(h * scale))

    original_resized = cv2.resize(original_cv, new_size, interpolation=cv2.INTER_LINEAR)
    enhanced_resized = cv2.resize(enhanced_cv, new_size, interpolation=cv2.INTER_LINEAR)

    # แสดงผลแยกหน้าต่าง
    cv2.imshow("Original Image", original_resized)
    cv2.imshow("Enhanced Image", enhanced_resized)
    cv2.waitKey(0)  # รอให้กดปุ่มใดๆ เพื่อปิดหน้าต่าง
    cv2.destroyAllWindows()
