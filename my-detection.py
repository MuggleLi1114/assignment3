
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
img_path = "/home/nvidia/jetson-inference/examples/my-recognition/brown_bear.jpg"
img = jetson.utils.loadImage(img_path)

detections = net.Detect(img)

for detection in detections:
	class_label  = net.GetClassDesc(detection.ClassID)
	confidence = detection.Confidence
	left = detection.Left
	top = detection.Top
	right = detection.Right
	bottom = detection.Bottom
	width = right - left
	height = bottom - top
	area = width * height
	center = ((right + left)/2, (bottom + top)/2)

print(f"class: {class_label}")
print(f"Confidence={confidence:.6f}")
print(f"Coordinates: Left={left}")
print(f"Coordinates: Top={top}")
print(f"Coordinates: Right={right}")
print(f"Coordinates: Bottom={bottom}")	
print(f"Coordinates: Width={width}")
print(f"Coordinates: Height={height}")
print(f"Coordinates: Area={area}")
print(f"Coordinates: Center={center}")
