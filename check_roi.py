import cv2
import matplotlib.pyplot as plt

IMAGE_PATH = "demoImage/demo.jpg"

# ===== 这里调 4 个百分比，直到框住水尺 =====
ROI_LEFT   = 0.30
ROI_RIGHT  = 0.70
ROI_TOP    = 0.10
ROI_BOTTOM = 0.85
# =========================================

img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

x1, x2 = int(w * ROI_LEFT),  int(w * ROI_RIGHT)
y1, y2 = int(h * ROI_TOP),   int(h * ROI_BOTTOM)

roi = img[y1:y2, x1:x2]

print(f"当前 ROI 百分比：L={ROI_LEFT}, R={ROI_RIGHT}, T={ROI_TOP}, B={ROI_BOTTOM}")
print(f"对应像素坐标：({x1}, {y1}) → ({x2}, {y2})")

# 在原图画框
img_box = img.copy()
cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示原图+框
plt.figure(figsize=(6,9))
plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)); plt.title("原图 + ROI 框"); plt.axis("off")
# 显示裁剪结果
plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)); plt.title("裁剪出的 ROI"); plt.axis("off")
plt.show()
