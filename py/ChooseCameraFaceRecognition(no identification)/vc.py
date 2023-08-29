print('Loading lib')
import cv2 as cv
# import numpy as np
print('Loading done')

# Camera OK flag
cameraOK = False
# Prompt user to choose camera (will correct for negative numbers and to large numbers)
camNum = abs(int(input("Choose camera: ")))%10


# Model from: https://github.com/kipr/opencv/tree/master/data/haarcascades
print('Loading face recognition model')
face_cascade = cv.CascadeClassifier('cascade/face.xml')
print('Loading done')

# Outer loop. Will run untill camera is connected or no camera can be found 
while not cameraOK and camNum >= 0:
    print('Connecting camera ', camNum)     
    
    # Create camera
    cam = cv.VideoCapture(camNum)

    # Inner loop. (given connected camera) Will run untill user press q 
    while True:
        # Get frame from camera and bool for success
        ok, frame = cam.read()
        
        if ok:
            # If cam.read() the camera is connected
            cameraOK = True
        else:
            # Else, reduce camNum and try connecting again
            camNum = camNum-1
            break

        # Convert frame to grayscale (for faster handling)
        fGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Detect faces in frame (returns x,y of starting point & width,height of the face (x,y,w,h))
        faces = face_cascade.detectMultiScale(fGray)
        # Looping in the case of multiple faces are detected
        for (x,y,w,h) in faces:
            # Draw a rectangle using coordinates of detected faces 
            cv.rectangle(frame,         # Were to draw the rectangle (on the frame used below) 
                         (x,y),         # Starting (upper left)
                         (x+w,y+h),     # End point (lower right)
                         (50, 255, 50), # Color
                         2)             # Thickness
        
        cv.imshow('test img',           # Title for the window (also used as id)
                  frame)                # What to show

        # Check for shutdown command from user
        if(cv.waitKey(1) == ord('q')):
            print('Shuttning down')
            break

# Shut down program
cam.release()
cv.destroyAllWindows()