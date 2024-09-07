import cv2
import numpy as np

# RTSP 스트리밍 주소 입력
rtsp_url = "rtsp://localhost:8554/stream"

weights_path = "C:\\Users\\weri0\\vsproject\\yolov4-tiny-custom_final.weights"
config_path = "C:\\Users\\weri0\\vsproject\\yolov4-tiny-custom.cfg"
names_path = "C:\\Users\\weri0\\vsproject\\_darknet.labels"

# YOLOv4-tiny 모델 불러오기
model = cv2.dnn.readNet(weights_path, config_path)

# 클래스 이름 불러오기
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

cap = cv2.VideoCapture(rtsp_url)

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        print("RTSP 스트림을 가져올 수 없습니다.")
        break

    # 이미지 전처리
    # 이미지를 416x416 크기로 변경
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)

    output_layers_names = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers_names)

    boxes = [] # 객체의 박스 좌표
    confidences = [] # 탐지된 객체의 신뢰도 
    class_ids = [] # 탐지된 객체의 클래스 ID

    for output in layer_outputs:
        for detection in output: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:  # 신뢰도가 (0.6)보다 큰 경우만 객체로 인식
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]) 
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 겹치는 객체의 BOX중 가장 신뢰도가 높은 BOX만 남김
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.6, nms_threshold=0.6)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            color = [0, 0, 255]  # 화재일 경우 빨간색 박스
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()