from __future__ import division
import cv2
import time
import numpy as np

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv2.imread("hand.jpg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
aspect_ratio = frameWidth/frameHeight

threshold = 0.1

t = time.time()
# input image dimensions for the network
inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
# print("time taken by network : {:.3f}".format(time.time() - t))

# Empty list to store the detected keypoints
points = []

#We are only intrested in these points for rings in the hand.

finger1 = 6
finger2 = 10
finger3 = 14
finger4 = 18

counter = 0


for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if counter == 6:

        if prob > threshold:



            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            # Add the point to the list if the probability is greater than the threshold

            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    if counter == 10:

        if prob > threshold:



            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            # Add the point to the list if the probability is greater than the threshold

            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    if counter == 14:

        if prob > threshold:


            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            # Add the point to the list if the probability is greater than the threshold

            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    if counter == 18:

        if prob > threshold:


            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            # Add the point to the list if the probability is greater than the threshold

            points.append((int(point[0]), int(point[1])))
        else:
            points.append(None)
    counter += 1



cv2.imshow('Output-Keypoints', frameCopy)

cv2.waitKey(0)

print("List of Finger Indexes" )
print(points)