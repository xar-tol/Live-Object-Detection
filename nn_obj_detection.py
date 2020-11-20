# Displays, using a neural network, identified objects in a webcam feed
# Can also replace the inside of line 9 VideoCapture() to identify objects in a video.

import numpy as np
import cv2

# load MobileNetSSD and startup video
model = cv2.dnn.readNetFromCaffe(r"MobileNetSSD.prototxt", r"MobileNetSSD.caffemodel")
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# grab class list for the model
with open("classList.txt") as file:
	class_list = file.read().splitlines()

# detect any objects in the video
while video.isOpened():
	# get current video frame and original dimensions
	_, frame = video.read()
	height, width = frame.shape[:2]

	# turn frame to blob and get predictions for the blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	model.setInput(blob)

	# detect objects
	objects = model.forward()

	# check the class of each object
	for i in np.arange(0, objects.shape[2]):
		currObj = objects[0, 0, i]

		# if it's relatively likely to be true, show it
		if currObj[2] >= 0.3:
			# get the outline for the detected object
			x1, x2 = int(currObj[3]*width), int(currObj[5]*width)
			y1, y2 = int(currObj[4]*height), int(currObj[6]*height)
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

			# add the detected object and label to the frame
			text = "{} - {:.0%}".format(class_list[int(currObj[1])], currObj[2])
			cv2.putText(frame, text, ((x1+x2)//2, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 0, 255), 2)

	# show output with detected objects
	cv2.imshow("Video with Objects", frame)

	# check for quit key
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q") or key == ord(" ") or key == 27:
		break

# close video and windows
video.release()
cv2.destroyAllWindows()