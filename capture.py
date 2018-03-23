import numpy as np
import cv2

video_name = 'eyes_avi.avi'

#Open the video file
cap = cv2.VideoCapture(video_name)

while(cap.isOpened()):
	
	# Capture frame-by-frame
	(grabbed, frame) = cap.read()

	# video
	if not grabbed:
		print("not grabbed")
		break
 
	
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()