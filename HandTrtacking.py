import cv2
import mediapipe as mp
import time

class handDetect():
    def __init__(self,mode = False,  max_hands = 2, modelcomplexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.modelcomplexity = modelcomplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        # mediapipe module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.modelcomplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots on hands total 20 landmarks point

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, detect):
        lmList = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    lmList.append([id, cx,cy])
                    if id == detect:
                        cv2.circle(img,(cx,cy),7,(255,0,0),cv2.FILLED)
        return lmList


