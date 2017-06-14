import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cv2


def showImg(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Takes a path input to an image frame
def get_corner(path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 300, apertureSize=3)

    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, 0.2, np.pi/180, 100, minLineLength, maxLineGap)
    

    if lines is None:
        return
    potentialCorner = {'x':[], 'y':[]}
    for line in lines:
        x1,y1,x2,y2 = line[0]

        # Track vertical and horizontal lines.
        if np.abs(0 - (y2 - y1)) < 0.5:
            potentialCorner['y'].append(y2)
        elif np.abs(0 - (x2 - x1)) < 0.5:
            potentialCorner['x'].append(x1)
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    # Assume any pair of vertical and horizontal lines form corners. 
    for x in potentialCorner['x']:
        for y in potentialCorner['y']:
            cv2.circle(img, (x, y), 5, (0, 255, ), 2)
            cv2.imwrite('hough/frame' + str(f) + '.jpg', img)



