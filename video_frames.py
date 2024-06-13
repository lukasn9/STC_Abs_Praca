import cv2
 
capture = cv2.VideoCapture("C:/Users/luna0/Downloads/VID20240529142023.mp4")
frameNum = 0
 
while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(f"C:/Users/luna0/Downloads/frames_capture/frame_{frameNum}.jpg", frame)
        print(f"Frame {frameNum}: Success")
 
    else:
        break
 
    frameNum += 1
capture.release()