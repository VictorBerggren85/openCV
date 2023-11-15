from threading import Thread as t
import cv2 as cv
import mediapipe as mp

class mpHands:
    def __init__(
            self, 
            maxHands=2, 
            init_min_detection_confidence=0.5, 
            init_min_tracking_confidence=0.5):
        self.hands=mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=init_min_detection_confidence,
            min_tracking_confidence=init_min_tracking_confidence)
    def Marks(self, frame, width=640, height=360):
        myHands=[]
        handIndexLabel=[]
        frameRGB=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for handLandmarks, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                myHand=[]
                handIndexLabel.append((str(hand.classification[0].index), str(hand.classification[0].label)))
                for landMark in handLandmarks.landmark:
                    myHand.append((int(landMark.x*width), int(landMark.y*height)))
                myHands.append(myHand)
        return handIndexLabel, myHands

class mpPose:
    def __init__(
            self,      
            init_static_image_mode=False,
            init_model_complexity=1,
            init_enable_segmentation=False,
            init_min_detection_confidence=0.5):
        self.poses=mp.solutions.pose.Pose(
            static_image_mode=init_static_image_mode,
            model_complexity=init_model_complexity,
            enable_segmentation=init_enable_segmentation,
            min_detection_confidence=init_min_detection_confidence)
    def Marks(self, frame, width=640, height=360):
        poses=[]
        frameRGB=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results=self.poses.process(frameRGB)
        if results.pose_landmarks != None:
            for lm in results.pose_landmarks.landmark:
                poses.append((int(lm.x*width), int(lm.y*height)))
        return poses        

class mpFace:
    def __init__(self):
        self.faces=mp.solutions.face_detection.FaceDetection()
    def Marks(self, frame, width=640, height=360):
        faces=[]
        frameRGB=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results=self.faces.process(frameRGB)
        if results.detections != None:
            for face in results.detections:
                box=face.location_data.relative_bounding_box
                faces.append((
                    (int(box.xmin*width), int(box.ymin*height)), 
                    (int((box.xmin+box.width)*width), int((box.ymin+box.height)*height))))
        return faces

class cvCap:
    def __init__(
            self, 
            camNum=0, 
            capWidth=640, 
            capHeight=360, 
            fps=30,
            codec=cv.VideoWriter_fourcc(*'MJPG')):
        self.cap=cv.VideoCapture(camNum, cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, capWidth)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, capHeight)
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.cap.set(cv.CAP_PROP_FOURCC, codec)
        self.frame=[]
        self.run=True
        # capThread=t(target=self.captureFrame, daemon=True)
        # capThread.run()
    def setSize(self, newWidth, newHeight):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, newWidth)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, newHeight)
    def captureFrame(self):
        while self.run:
            self.frame=self.cap.read()
    def getFrame(self):
        return self.frame
    def destroy(self):
        self.run=False
        self.cap.release()
