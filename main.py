from ultralytics import YOLO

import cv2
import torch
from PIL import Image
import numpy as np
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')
#model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')


# results = model('https://ultralytics.com/images/bus.jpg')

# #results = model.predict(image_path)

# for result in results:
#     print(result.boxes)
#     print(result.probs)
#     result.save("processed_yolo.jpg")
#     result.show()

# Set the model to evaluation mode
results = model(source=0, show=True, conf=0.4, save=True)

# Initialize the webcam
# cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert the image from BGR (OpenCV format) to RGB
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         # Inference
#         results = model.predict(img, size=640)  # Resize to match the model's required input size

#         # Convert results to original frame for displaying
#         results.render()  # Updates results.imgs with boxes and labels
#         annotated_image = cv2.cvtColor(np.array(results.imgs[0]), cv2.COLOR_RGB2BGR)

#         # Show the image
#         cv2.imshow('YOLOv8 Real-Time Detection', annotated_image)
#         if cv2.waitKey(1) == ord('q'):  # Exit on pressing 'q'
#             break
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
