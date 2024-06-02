from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("sample.mp4")

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("track_result.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

track_history = defaultdict(lambda: [])

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )

        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
    else:
        break

cap.release()
cv2.destroyAllWindows()
