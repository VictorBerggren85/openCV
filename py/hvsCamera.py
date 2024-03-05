from threading import Thread as t
from cv2 import (
    cvtColor,putText,morphologyEx,findContours,contourArea,
    boundingRect,inRange,bitwise_or,VideoWriter_fourcc,VideoCapture,
    CAP_DSHOW,CAP_PROP_FRAME_WIDTH,CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FPS,CAP_PROP_FOURCC,FONT_HERSHEY_COMPLEX,
    COLOR_RGB2BGR,MORPH_OPEN,MORPH_CLOSE,CHAIN_APPROX_SIMPLE,
    RETR_EXTERNAL,COLOR_BGR2HSV)
from numpy import array,uint8,ones,zeros
from math import sqrt

DEBUGGING=True
if DEBUGGING:
    import cv2 as cv

class Cam:

    def __init__(
            self,
            camNum=0,
            capWidth=640,
            capHeight=360,
            fps=30,
            codec=VideoWriter_fourcc(*'MJPG')):
        self.cap=VideoCapture(camNum,CAP_DSHOW)
        self.height=capHeight
        self.width=capWidth
        self.longestPossibleOffset=sqrt(pow(capWidth*.5,2)+pow(capHeight*.5,2))
        self.globalCenter=(int(capWidth*.5),int(capHeight*.5))
        self.cap.set(CAP_PROP_FRAME_WIDTH,capWidth)
        self.cap.set(CAP_PROP_FRAME_HEIGHT,capHeight)
        self.cap.set(CAP_PROP_FPS,fps)
        self.cap.set(CAP_PROP_FOURCC,codec)
        self.frame=[]
        self.greenMask=array([])
        self.greenCenter=(0,0)
        self.redMask=array([])
        self.redCenter=(0,0)
        self.blueMask=array([])
        self.blueCenter=(0,0)
        self.yellowMask=array([])
        self.yellowCenter=(0,0)
        self.kernel=ones((5,5),uint8)
        self.readSuccess=False
        self.createMasks=False
        self.emptyFrame=zeros((capWidth,capHeight),uint8)
        putText(
            self.emptyFrame,'Error reading frame',(25,25), 
            FONT_HERSHEY_COMPLEX,1,(255,255,255), 1)
        self.run=True
        self.capThread=t(target=self.captureFrame,args=(),daemon=True)
        # self.cvtThread=t(target=self.createColorMasks,args=(),daemon=True)
        # self.cvtThread.start()
        self.capThread.start()
        print('Width:\t'+str(self.cap.get(CAP_PROP_FRAME_WIDTH)))
        print('Height:\t'+str(self.cap.get(CAP_PROP_FRAME_HEIGHT)))

    def setMaskCreation(self,createMask):
        self.createMasks=createMask

    def setSize(self,newWidth,newHeight):
        self.cap.set(CAP_PROP_FRAME_WIDTH,newWidth)
        self.cap.set(CAP_PROP_FRAME_HEIGHT,newHeight)

    def getFrame(self):
        if self.readSuccess:
            return self.frame
        else:
            return cvtColor(self.emptyFrame,COLOR_RGB2BGR)

    def getFrameSize(self):
        return (self.cap.get(CAP_PROP_FRAME_WIDTH),self.cap.get(CAP_PROP_FRAME_HEIGHT))

    # Can only be used after createMasks have been set to True (using function setMaskCreation)
    def getRedMask(self):
        self.redMask=morphologyEx(src=self.redMask,
                                     op=MORPH_OPEN,
                                     kernel=self.kernel)
        return morphologyEx(self.redMask,MORPH_CLOSE,self.kernel)
    def getRedCenter(self):
        return self.redCenter

    # Can only be used after createMasks have been set to True (using function setMaskCreation)
    def getGreenMask(self):
        self.greenMask=morphologyEx(src=self.greenMask,
                                       op=MORPH_OPEN,
                                       kernel=self.kernel)
        return morphologyEx(self.greenMask,MORPH_CLOSE,self.kernel)
    def getGreenCenter(self):
        return self.greenCenter

    # Can only be used after createMasks have been set to True (using function setMaskCreation)
    def getBlueMask(self):
        self.blueMask=morphologyEx(src=self.blueMask,
                                       op=MORPH_OPEN,
                                       kernel=self.kernel)
        return morphologyEx(self.blueMask,MORPH_CLOSE,self.kernel)
    def getBlueCenter(self):
        return self.blueCenter
    # Can only be used after createMasks have been set to True (using function setMaskCreation)
    def getYellowMask(self):
        self.yellowMask=morphologyEx(src=self.yellowMask,
                                       op=MORPH_OPEN,
                                       kernel=self.kernel)
        return morphologyEx(self.yellowMask,MORPH_CLOSE,self.kernel)
    def getYellowCenter(self):
        return self.yellowCenter

    #Search for object in mask, calculates center of object using blounding box coordinates.
    #Returns center coordinates of largest object and offset from view (global) center.
    #If no objet is found coordinates and offset will by default be outside of view
    def findCenter(self,mask):
        contours,_=findContours(
            mask,
            RETR_EXTERNAL,       #Find only external details
            CHAIN_APPROX_SIMPLE) #Compress contours to endpoints (x,y,w,h)
        largest_area=0

        #Coordinates for obj bounding box (starts outside the view)
        x,y,rw,rh=(self.width*10,self.height*10,0,0)
        
        if len(contours) > 0:
            #Use contours to parse out closest object in mask
            for c in contours:
                area=contourArea(c)
                if area>largest_area:
                    x,y,rw,rh=boundingRect(c)
                    largest_area=area
        #Calculate return values 
        center=x+int(rw*.5),y+int(rh*.5)
        offset=int(sqrt(pow(center[0],2)+pow(center[1],2))-
                sqrt(pow(self.globalCenter[0],2)+pow(self.globalCenter[1],2)))
        return(
            (center),       #Object center point
            (offset),       #Offset from global center 
            (largest_area), #Object area
            (rw,rh))
    def createColorMasks(self):
        # while self.run:
        if self.readSuccess:
            hsv=cvtColor(self.frame,COLOR_BGR2HSV)
            self.greenMask=inRange(
                hsv,
                array([35,55,55]),
                array([75,255,255]))
            self.greenCenter=self.findCenter(self.greenMask)
            
            self.blueMask=inRange(
                hsv,
                array([90,0,0]),
                array([125,255,255]))
            self.blueCenter=self.findCenter(self.blueMask)
            
            self.yellowMask=inRange(
                hsv,
                array([29,128,127]),
                array([35,255,255]))
            self.yellowCenter=self.findCenter(self.yellowMask)
            
            self.redMask=bitwise_or(
                inRange(
                    hsv,
                    array([0, 75, 70]),
                    array([10, 255, 255])),
                inRange(
                    hsv,
                    array([170, 75, 70]),
                    array([180, 255, 255])))
            self.redCenter=self.findCenter(self.redMask)

    def captureFrame(self):
        while self.run:
            self.readSuccess,self.frame=self.cap.read()
            if self.createMasks:
                self.createColorMasks()

    def destroy(self):
        self.capThread=0
        # self.cvtThread=0
        self.run=False
        self.cap.release()


class Runner:
    from enum import Enum
    from time import sleep
    class Mode(Enum):
        IDLE=0
        SEARCH1=1
        FETCH=2
        SEARCH2=3
        DEPOSIT=4
        VERIFY=5
    class Target(Enum):
        RED=0
        GREEN=1
        BLUE=2
        YELLOW=3

    def __init__(self):
        if DEBUGGING:
            self.debugFrame=[]
        self.cam=Cam()
        self.rotation=(
            0,  #Left num steps
            0)  #Right num steps
        self.numUnsorted=12
        self.numSorted=0
        self.target=None
        self.state=self.Mode.IDLE
        self.startSignal=False
        self.runThread=t(target=self.run,args=(),daemon=True)
        for _ in range(5):
            if self.cam.readSuccess:
                self.cam.setMaskCreation(True)
                break
            else:
                self.sleep(.2)
        if self.cam.readSuccess:
            self.runThread.start()
        else:
            print('Failed to connect camera')
            self.destroy()

    def setStartSignal(self,newSignal):
        self.startSignal=newSignal

    def setState(self,newState):
        self.state=newState

    def getState(self):
        return self.state
        
    def calibrate(self):
        #Calculate 1 rotation
        pass

    def zeroIn(self,obj,goal):
        delta=goal-obj
        if abs(delta)<10:
            print('go')
        else:
            if delta<0:
                print('right '+str(abs(delta)))
            else:
                print('left '+str(delta))

    #Main run sequence
    def run(self):
        while True:
            if self.state==self.Mode.IDLE:
                self.idle()
            if self.state==self.Mode.SEARCH1:
                self.search1()
            if self.state==self.Mode.FETCH:
                self.fetch()
            if self.state==self.Mode.SEARCH2:
                self.search2()
            if self.state==self.Mode.DEPOSIT:
                self.deposit()
            if self.state==self.Mode.VERIFY:
                self.verify()

    #Do nothing, wait for start signal
    def idle(self):
        print('IDLE')
        goal=True
        while self.state==self.Mode.IDLE:
            result=self.startSignal
            if result==goal:
                break
        self.setState(self.Mode.SEARCH1)

    #Find objective. If none in view, rotate
    def search1(self):
        print('SEARCH1')
        
        centerPoint=0   #index
        offset=1        #index
        area=2          #index
        goal=True
        result=False

        while self.state==self.Mode.SEARCH1:
            rC=self.cam.redCenter
            gC=self.cam.greenCenter

            #Select target
            #If several targets in view
            if ((rC[centerPoint])[0]<self.cam.width and   #Check if x coordinate in view
                (gC[centerPoint])[0]<self.cam.width):     #Check if x coordinate in view
                #Compare area size (proximity)
                if rC[area]>gC[area]:
                    self.target=self.Target.RED
                    if DEBUGGING: 
                        frame=self.cam.getFrame()
                        cv.circle(frame,rC[0],10,(0,0,255),-1)
                        self.debugFrame=frame
                elif rC[area]<gC[area]:                 
                    self.target=self.Target.GREEN
                    if DEBUGGING: 
                        frame=self.cam.getFrame()
                        cv.circle(frame,gC[0],10,(0,255,0),-1)
                        self.debugFrame=frame
                #If equal size, compare offset
                else:
                    if rC[offset]<gC[offset]:
                        self.target=self.Target.RED
                        if DEBUGGING: 
                            frame=self.cam.getFrame()
                            cv.circle(frame,rC[0],10,(0,0,255),-1)
                            self.debugFrame=frame
                    elif rC[offset]>gC[offset]:
                        self.target=self.Target.GREEN
                        if DEBUGGING: 
                            frame=self.cam.getFrame()
                            cv.circle(frame,gC[0],10,(0,255,0),-1)
                            self.debugFrame=frame
                    #If bouth objects are equal in size and offset, select red
                    else:
                        self.target=self.Target.RED
                        if DEBUGGING: 
                            frame=self.cam.getFrame()
                            cv.circle(frame,rC[0],10,(0,0,255),-1)
                            self.debugFrame=frame

            #Only red object in view
            elif ((rC[centerPoint])[0]<self.cam.width and #Check if x coordinate in view
                (gC[centerPoint])[0]>self.cam.width):     #Check if x coordinate in view
                self.target=self.Target.RED
                if DEBUGGING: 
                    frame=self.cam.getFrame()
                    cv.circle(frame,rC[0],10,(0,0,255),-1)
                    self.debugFrame=frame
            #Only green object in view
            elif ((rC[centerPoint])[0]>self.cam.width and #Check if x coordinate in view
                (gC[centerPoint])[0]<self.cam.width):     #Check if x coordinate in view
                self.target=self.Target.GREEN
                if DEBUGGING: 
                    frame=self.cam.getFrame()
                    cv.circle(frame,gC[0],10,(0,255,0),-1)
                    self.debugFrame=frame

            #If no targets in view
            #Search for target
            else:
                print('No target: Turn RIGHT')
                self.rotation=(self.rotation[0]+10,self.rotation[1])
            
            result=self.target!=None
            if result==goal:
                break
        self.setState(self.Mode.FETCH)

    #Center on and go to object
    def fetch(self):
        print('FETCH')
        print('target='+str(self.target))

        x=0     #Index
        y=1     #Index
        goal=True
        height=self.cam.height
        roiSize=(150,70)

        while self.state==self.Mode.FETCH:
            if self.target==self.Target.RED:
                centerPoint,_,_,targetSize=self.cam.getRedCenter()
            elif self.target==self.Target.GREEN:
                centerPoint,_,_,targetSize=self.cam.getGreenCenter()
            else:
                print('ERROR! fetch state: unknown target')
            #If target is lost, return to search1 state
            if (centerPoint[0]>self.cam.width or
                targetSize[0]<=0 or
                targetSize[1]<=0):
                self.state=self.Mode.SEARCH1
                self.target=None
            
            self.zeroIn(centerPoint[x],self.cam.globalCenter[x])

            #Result ok if target inside of roi
            result=(centerPoint[x]+int(targetSize[x]*.5)<self.cam.globalCenter[x]+int(roiSize[x]*.5) and
                    centerPoint[x]-int(targetSize[x]*.5)>self.cam.globalCenter[x]-int(roiSize[x]*.5) and
                    centerPoint[y]+int(targetSize[y]*.5)<height and
                    centerPoint[y]-int(targetSize[y]*.5)>height-roiSize[y])
            
            if result==goal:
                break

        #Check if mode was changed (wich indicates lost target)
        if self.state==self.Mode.FETCH:
            self.setState(self.Mode.SEARCH2)
            if self.target==self.Target.RED:
                self.target=self.Target.YELLOW
            elif self.target==self.Target.GREEN:
                self.target=self.Target.BLUE

    #Use self.rotation to calculate where to deposit object
    #find deposit color based on object color
    def search2(self):
        print('SEARCH2')

        x=0             #Index
        y=1             #Index
        leftMotor=0     #Index
        rightMotor=1    #Index
        acceptableOffset=10
        goal=True
        move=(self.rotation[0]*-1,self.rotation[1]*-1)

        while self.state==self.Mode.SEARCH2:
            if self.target==self.Target.BLUE:
                centerPoint,_,_,targetSize=self.cam.getBlueCenter()
            elif self.target==self.Target.YELLOW:
                centerPoint,_,_,targetSize=self.cam.getYellowCenter()
            else:
                print('ERROR! search2 state: unknown target')

            #Check if deposit is in view
            if targetSize[x]<=0 or centerPoint[x]>self.cam.width:
                if move[leftMotor]>move[rightMotor]:
                    print('Deposit not in view, move right')
                else:
                    print('Deposit not in view, move left')
            else:
                self.zeroIn(centerPoint[x],self.cam.globalCenter[x])

            result=(centerPoint[x]>self.cam.globalCenter[x]-acceptableOffset and
                    centerPoint[x]<self.cam.globalCenter[x]+acceptableOffset)
            if result==goal:
                break
        self.setState(self.Mode.DEPOSIT)

    #Move to deposit, back off
    def deposit(self):
        print('DEPOSIT')

        goal=True

        while self.state==self.Mode.DEPOSIT:
            pass
            # if result==goal:
            #     self.numUnsorted=self.numUnsorted-1
            #     self.numSorted=self.numSorted+1
            #     break
        self.setState(self.Mode.DEPOSIT)

    #Check if objectiva has been achieved
    def verify(self):
        print('VERIFY')
        goal=12
        while self.state==self.Mode.VERIFY:
            result=self.numSorted
            if result==goal:
                self.startSignal=False
                self.setState(self.Mode.IDLE)
            else:
                self.setState(self.Mode.SEARCH1)

    def destroy(self):
        self.runThread=0
        self.cam.destroy()
