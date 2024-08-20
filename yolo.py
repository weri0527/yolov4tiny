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

    # YOLO 입력 크기 지정
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)

    output_layers_names = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 임계값 조정 가능
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maxima Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

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