import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("1.png")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, result = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
contour, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) ### WE can use cv.CHAIN_APPROX_NONE and cv.CHAIN_APPROX_SIMPLE to contour the letter completely or certain points

for i in range(len(contour)):
    cnt = contour[i]
    img = cv.drawContours(img, cnt, -1, (0, 255, 0), 4)
    ### کشیدن مستطیلی که شکل رو احاطه میکنه :
    x,y,w,h = cv.boundingRect(cnt)
    img = cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
    # contour area for finding dots .
    print(cv.contourArea(cnt))
    # Croping and saving:
    crop = img[y:y+h, x:x+w]
    cv.imwrite(f"/data/crop{i}.jpg",crop)
    cv.imshow("Image", img)
    cv.imshow("Crop", crop)
    cv.waitKey(0)