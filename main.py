import handmodule1
import  cv2
import time
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = handmodule1.handDetect()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    detector.findPosition(img,5)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70),cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

