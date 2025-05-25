import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---------------- 路径和参数 ----------------
IMAGE_PATH      = "demoImage/MVIMG_20250519_172912.jpg"  # 你的图片路径
WATERLINE_MODEL = "waterline.pt"               # 水面线模型
DIGIT_MODEL     = "train8/weights/best.pt"     # 数字+E刻度模型
ROI_HEIGHT_LIST = [400, 800, 1200, 1600, 2000] # ROI高度尝试
CONF_LINE       = 0.25
CONF_DIGIT      = 0.15
# -------------------------------------------

def show(img, title=""):
    plt.figure(figsize=(6, 9))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontproperties="SimHei")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def get_main_digit_anchors(boxes, y1_offset=0):
    main_digits = []
    aux_digits = []
    for box in boxes:
        cls = int(box.cls[0].cpu().numpy())
        xyxy = box.xyxy[0].cpu().numpy()
        y_bottom = y1_offset + xyxy[3]
        if cls == 10:  # “E” 代表 5cm
            aux_digits.append((5, y_bottom))
        else:
            main_digits.append((cls * 10, y_bottom))
    return main_digits, aux_digits

def interpolate_by_digits(anchors, water_y):
    anchors.sort(key=lambda x: x[1])
    depth_cm = None
    for i in range(len(anchors) - 1):
        v0, y0 = anchors[i]
        v1, y1 = anchors[i + 1]
        if y0 <= water_y <= y1 or y1 <= water_y <= y0:
            depth_cm = v0 + (v1 - v0) * (water_y - y0) / (y1 - y0)
            break
    if depth_cm is None and len(anchors) >= 2:
        if water_y < anchors[0][1]:
            v0, y0 = anchors[0]
            v1, y1 = anchors[1]
            cm_per_px = (v1 - v0) / (y1 - y0)
            depth_cm = v0 + (water_y - y0) * cm_per_px
        elif water_y > anchors[-1][1]:
            v0, y0 = anchors[-2]
            v1, y1 = anchors[-1]
            cm_per_px = (v1 - v0) / (y1 - y0)
            depth_cm = v1 + (water_y - y1) * cm_per_px
    return depth_cm

def main():
    if not os.path.exists(IMAGE_PATH):
        print("❌ 找不到图片")
        return
    img = cv2.imread(IMAGE_PATH)
    h, w = img.shape[:2]

    wl_model = YOLO(WATERLINE_MODEL)
    wl_res = wl_model.predict(img, conf=CONF_LINE, verbose=False)
    if len(wl_res[0].boxes) == 0:
        print("❌ 未检测到水平面")
        return
    xyxy_wl = max(wl_res[0].boxes.xyxy.cpu().numpy(), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    water_y = (xyxy_wl[1] + xyxy_wl[3]) / 2
    cv2.rectangle(img, (int(xyxy_wl[0]), int(xyxy_wl[1])), (int(xyxy_wl[2]), int(xyxy_wl[3])), (0, 0, 255), 2)

    dg_model = YOLO(DIGIT_MODEL)
    main_digits, aux_digits = [], []
    for roi_h in ROI_HEIGHT_LIST:
        y1 = max(0, int(water_y - roi_h // 2))
        y2 = min(h, int(water_y + roi_h // 2))
        roi = img[y1:y2].copy()
        dg_res = dg_model.predict(roi, conf=CONF_DIGIT, verbose=False)
        main_digits, aux_digits = get_main_digit_anchors(dg_res[0].boxes, y1_offset=y1)
        if len(main_digits) >= 2:
            break

    if len(main_digits) < 2:
        dg_res = dg_model.predict(img, conf=CONF_DIGIT, verbose=False)
        main_digits, aux_digits = get_main_digit_anchors(dg_res[0].boxes)

    if len(main_digits) >= 2:
        anchors = main_digits
    elif len(main_digits) == 1 and len(aux_digits) >= 1:
        print("⚠️ 主数字不足，启用 E（5cm）辅助计算")
        anchors = main_digits + aux_digits
    else:
        print("❌ 无法计算水深")
        show(img, "仅水平面")
        return

    for v, y in anchors:
        cv2.circle(img, (w // 2, int(y)), 8, (0, 255, 0), -1)
        cv2.putText(img, str(int(v)), (w // 2 + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    depth_cm = interpolate_by_digits(anchors, water_y)
    print("anchors(cm, y):", anchors)
    print(f"\u26f2 推算水深约：{depth_cm:.1f} cm")

   # return depth_cm

    cv2.putText(img, f"Depth: {depth_cm:.1f} cm", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    show(img, "主数字+E辅助测深（误差±1cm）")

    return depth_cm

if __name__ == "__main__":
    main()
