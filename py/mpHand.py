class mpHands:
    import mediapipe as mp
    import cv2 as cv
    def __init__(
            self, 
            maxHands=2, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5):
        self.hands=self.mp.solutions.hands.Hands(
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
    def Marks(self, frame, width=640, height=360):
        myHands=[]
        frameRGB=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                myHand=[]
                for landMark in handLandmarks.landmark:
                    myHand.append((int(landMark.x*width), int(landMark.y*height)))
                myHands.append(myHand)
        return myHands
