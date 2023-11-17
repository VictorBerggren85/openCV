# from threading import Thread as t
import cv2 as cv
# import mediapipe as mp
from mediapipe import solutions as mps

class mpHands:
    def __init__(
            self, 
            maxHands=2, 
            init_min_detection_confidence=0.5, 
            init_min_tracking_confidence=0.5):
        # self.hands=mp.solutions.hands.Hands(
        self.hands=mps.hands.Hands(
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
        # self.poses=mp.solutions.pose.Pose(
        self.poses=mps.pose.Pose(
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
        # self.faces=mp.solutions.face_detection.FaceDetection()
        self.faces=mps.face_detection.FaceDetection()
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

class mpFaceMesh:
    def __init__(
            self,
            init_static_image_mode=True,
            init_max_num_faces=1,
            init_refine_landmarks=True,
            init_min_detection_confidence=0.5):
        self.faceMesh=mps.face_mesh.FaceMesh(
            static_image_mode=init_static_image_mode,
            max_num_faces=init_max_num_faces,
            refine_landmarks=init_refine_landmarks,
            min_detection_confidence=init_min_detection_confidence)
    def Marks(self, frame, width=640, height=360):
        MeshLandMarks=[]
        frameRGB=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results=self.faceMesh.process(frameRGB)
        if results.multi_face_landmarks != None:
            for faceLM in results.multi_face_landmarks:
                for lm in faceLM.landmark:
                    MeshLandMarks.append((int(lm.x*width), int(lm.y*height)))
        return MeshLandMarks
        
class cvCap:
    from threading import Thread as t
    import numpy as np
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
        # self.width=capWidth
        # self.height=capHeight
        self.frame=[]
        self.readSuccess=False
        self.emptyFrame=self.np.zeros((capWidth, capHeight), self.np.uint8)
        cv.putText(
            self.emptyFrame, 'Error reading frame', (25,25), 
            cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        self.run=True
        capThread=self.t(target=self.captureFrame, args=(), daemon=True)
        capThread.start()
    def setSize(self, newWidth, newHeight):
        # self.width=newWidth
        # self.height=newHeight
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, newWidth)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, newHeight)
    def captureFrame(self):
        while self.run:
            self.readSuccess, self.frame=self.cap.read()
    def getFrame(self):
        if self.readSuccess:
            return self.frame
        else:
            return cv.cvtColor(self.emptyFrame, cv.COLOR_RGB2BGR)
    def destroy(self):
        self.run=False
        self.cap.release()