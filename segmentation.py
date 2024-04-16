
from ultralytics import YOLO
model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')


# results = model('https://ultralytics.com/images/bus.jpg')

# #results = model.predict(image_path)

# for result in results:
#     print(result.boxes)
#     print(result.probs)
#     result.save("processed_yolo.jpg")
#     result.show()

# Set the model to evaluation mode
results = model(source=0, show=True, conf=0.4, save=True)