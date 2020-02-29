#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
from centroidClass import CentroidTracker


def nonMaxSuppression(boxes, overlapThresh):
    """Implements the nonmax suppression Algorithm
    Suppresses the overlapped contours into a single one.

    Args:
        boxes: An array with the boxes of shape [N_boxes, x, y, w, h]
        overlapThresh: The threshold to suppress the overlapped boxes

    Returns:
        boxes: An array with the suppressed boxes
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs,
                np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype('int')


def getCentroids(boxes, frameWidth):
    """Given a list of rectangles and the width of the frame computes
    and returns the centroids for in each lane. We split the centroids in
    order to reduce the number of comparisons between centroids later.

    Args:
        boxes: The bounding boxes coordinates in (x, y, w, h)
    Returns:
        leftLaneCentroids: A list with the centroids of the boxes on the left
            lane
        right:LaneCentroids: A list with the centroids of the boxes on the
            right lane.
    """

    leftLaneCentroids = []
    rightLaneCentroids = []

    for (x, y, w, h) in boxes:
        centroid = [x + w // 2, y + h // 2]
        if centroid[0] < frameWidth // 2:
            leftLaneCentroids.append(centroid)
        else:
            rightLaneCentroids.append(centroid)
    return leftLaneCentroids, rightLaneCentroids


# ------------------ Main ------------------------------ #
def main(videoFile, outputFile):
    """Main Function of the program
    """

    if not os.path.exists(videoFile):
        print('The following path does not exist {}'.format(videoFile))
        sys.exit(-1)

    video = cv2.VideoCapture(videoFile)

    # Initialize some counters for frames. Useful to compute the
    # average of frames.
    frameCounter = 0
    sumFps = 0
    avgFps = 0

    # Read the first frame
    ok, firstFrame = video.read()
    if not ok:
        print("Error reading frame...")
        sys.exit(-1)

    firstFrame = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
    firstFrame = cv2.GaussianBlur(firstFrame, (17, 17), 0)

    # Frame dimensions
    height, width = firstFrame.shape

    # Left and right lane window containing the top left and bottom right
    # edges (x1, y1, x2, y2)
    leftLaneWindow = (100, 480, 550, 600)
    rightLaneWindow = (730, 400, 1220, 600)

    # Lane counters for each current
    leftLaneCounter = 0
    rightLaneCounter = 0

    # Instantiate the 2 centroid Trackers
    leftObjectTracker = CentroidTracker(leftLaneWindow)
    rightObjectTracker = CentroidTracker(rightLaneWindow)

    # Define the codec and create VideoWriter object
    vidWriter = cv2.VideoWriter(outputFile,
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          30,
                          (width, height))

    while True:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                print("Exit ")
                break

            # Start timer
            timer = cv2.getTickCount()

            # Process
            blurred = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(blurred, (17, 17), 0)

            # Compute the difference between 2 consecutive frames and capture
            # the moveness
            frameDelta = cv2.absdiff(firstFrame, blurred)
            thresh = cv2.threshold(frameDelta, 15, 255, cv2.THRESH_BINARY)[1]

            # Dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=5)
            contours = cv2.findContours(
                    thresh.copy(),
                    cv2.RETR_CCOMP,
                    cv2.CHAIN_APPROX_SIMPLE)[1]

            # Array with the contours over certain area
            filteredContours = np.array([])

            # Loop over the contours
            for contour in contours:
                # If the contour is too small, ignore it
                if cv2.contourArea(contour) < 1200:
                    continue
                if filteredContours.size == 0:
                    filteredContours = np.expand_dims(cv2.boundingRect(contour), axis=0)
                else:
                    filteredContours = np.concatenate(
                            (filteredContours, np.expand_dims(cv2.boundingRect(contour), axis=0)),
                            axis=0)

            # Suppress the overlapped boxes
            suppressedContours = nonMaxSuppression(filteredContours, overlapThresh=0.2)
            for (x, y, w, h) in suppressedContours:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get the centroids of the boxes in the current frame in two lists
            # for each current respectively.
            leftCentroids, rightCentroids = getCentroids(suppressedContours, width)
            # print(rightCentroids)

            # Update the centroids
            leftObjectTracker.update(leftCentroids, frameCounter)
            leftLaneCounter = leftObjectTracker.getVehicleCounter()
            rightObjectTracker.update(rightCentroids, frameCounter)
            rightLaneCounter = rightObjectTracker.getVehicleCounter()

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame and the car counts
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "Vehicles: " + str(int(leftLaneCounter)), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 150), 2)
            cv2.putText(frame, "Vehicles: " + str(int(rightLaneCounter)), (1000, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 150), 2)


            # Display result
            # cv2.rectangle(frame, leftLaneWindow[0:2], leftLaneWindow[2:4], (0, 0, 255), 2)
            # cv2.rectangle(frame, rightLaneWindow[0:2], rightLaneWindow[2:4], (0, 0, 255), 2)
            cv2.imshow("Video", frame)

            # Pass the previous frame
            firstFrame = blurred

            # Compute the average fps
            frameCounter += 1
            sumFps += fps
            avgFps= sumFps / frameCounter

            vidWriter.write(frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

    vidWriter.release()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Wrong number of arguments')
        sys.exit(-1)
    videoFile = sys.argv[1]
    outputFile = sys.argv[2]

    if not outputFile.endswith('.avi'):
        print('Incorrect output file type. It should be -> .avi ')
        sys.exit(-1)
    main(videoFile, outputFile)
