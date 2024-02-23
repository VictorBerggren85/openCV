import cv2 as cv

class Cam:
    from threading import Thread as t
    import numpy as np
    def __init__(
            self, 
            camNum=0, 
            capWidth=640, 
            capHeight=360, 
            fps=30,
            codec=cv.VideoWriter_fourcc(*'MJPG')):
        self.cap=cv.VideoCapture(camNum,cv.CAP_DSHOW)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH,capWidth)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT,capHeight)
        self.cap.set(cv.CAP_PROP_FPS,fps)
        self.cap.set(cv.CAP_PROP_FOURCC,codec)
        self.frame=[]
        self.greenMask=self.np.array([])
        self.redMask=self.np.array([])
        self.blueMask=self.np.array([])
        self.yellowMask=self.np.array([])
        self.kernel=self.np.ones((5,5),self.np.uint8)
        self.readSuccess=False
        self.emptyFrame=self.np.zeros((capWidth,capHeight),self.np.uint8)
        cv.putText(
            self.emptyFrame,'Error reading frame',(25,25), 
            cv.FONT_HERSHEY_COMPLEX,1,(255,255,255), 1)
        self.run=True
        self.capThread=self.t(target=self.captureFrame,args=(),daemon=True)
        # self.cvtThread=self.t(target=self.createColorMasks,args=(),daemon=True)
        # self.cvtThread.start()
        self.capThread.start()
    
    def createColorMasks(self):
        # while self.run:
        if self.readSuccess:
            print('cvt')
            hsv=cv.cvtColor(self.frame,cv.COLOR_BGR2HSV)
            self.greenMask=cv.inRange(
                hsv,
                self.np.array([35,55,55]),
                self.np.array([70,255,255]))
            self.blueMask=cv.inRange(
                hsv,
                self.np.array([90,0,0]),
                self.np.array([120,255,255]))
            self.yellowMask=cv.inRange(
                hsv,
                self.np.array([29,128,127]),
                self.np.array([35,255,255]))
            self.redMask=cv.bitwise_or(
                cv.inRange(
                    hsv,
                    self.np.array([0, 75, 70]),
                    self.np.array([10, 255, 255])),
                cv.inRange(
                    hsv,
                    self.np.array([170, 75, 70]),
                    self.np.array([180, 255, 255])))

    def setSize(self,newWidth,newHeight):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH,newWidth)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT,newHeight)

    def captureFrame(self):
        while self.run:
            print('cap')
            self.readSuccess,self.frame=self.cap.read()
            self.createColorMasks()


    def getFrame(self):
        if self.readSuccess:
            return self.frame
        else:
            return cv.cvtColor(self.emptyFrame,cv.COLOR_RGB2BGR)
        
    def getRedMask(self):
        self.redMask=cv.morphologyEx(src=self.redMask,
                                     op=cv.MORPH_OPEN,
                                     kernel=self.kernel)
        return cv.morphologyEx(self.redMask,cv.MORPH_CLOSE,self.kernel)
    
    def getGreenMask(self):
        self.greenMask=cv.morphologyEx(src=self.greenMask,
                                       op=cv.MORPH_OPEN,
                                       kernel=self.kernel)
        return cv.morphologyEx(self.greenMask,cv.MORPH_CLOSE,self.kernel)
    
    def getBlueMask(self):
        self.blueMask=cv.morphologyEx(src=self.blueMask,
                                       op=cv.MORPH_OPEN,
                                       kernel=self.kernel)
        return cv.morphologyEx(self.blueMask,cv.MORPH_CLOSE,self.kernel)
    
    def getYellowMask(self):
        self.yellowMask=cv.morphologyEx(src=self.yellowMask,
                                       op=cv.MORPH_OPEN,
                                       kernel=self.kernel)
        return cv.morphologyEx(self.yellowMask,cv.MORPH_CLOSE,self.kernel)
    
    def destroy(self):
        self.capThread=0
        self.cvtThread=0
        self.run=False
        self.cap.release()
