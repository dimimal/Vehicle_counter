#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker():
    """Implements the tracker Class for tracking the centroids from each
    individual stream.
    """
    def __init__(self, window, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared" respectively
        self.window = window
        self.nextObjectID = 0
        self.vehicleCounter = 0
        self.maxDisappeared = maxDisappeared

        # Define the dictionarys holding the centroids wich are valid
        # and those that cannot be tracked anymore along with the frame
        # that have been tracked
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.objectFramePick = OrderedDict()

    def register(self, centroids):
        # when registering an object we use the next available object
        # ID to store the centroid
        if not isinstance(centroids[0], list):
            self.objects[self.nextObjectID] = centroids
            self.disappeared[self.nextObjectID] = 0
            self.objectFramePick[self.nextObjectID] = self.currentFrame
            self.nextObjectID += 1
        else:
            for centroid in centroids:
                self.objects[self.nextObjectID] = centroid.flatten()
                self.disappeared[self.nextObjectID] = 0
                self.objectFramePick[self.nextObjectID] = self.currentFrame
                self.nextObjectID += 1

    def update(self, centroids, curFrame):
        # Set the frame to current frame instance
        self.currentFrame = curFrame
        # check to see if the list of centroids is empty
        if len(centroids) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            copyDisappeared = self.disappeared.copy()
            for objectID in copyDisappeared.keys():
                self.disappeared[objectID] += 1

                # If we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deletete it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Update the counter
            self.updateVehicleCounter()

            # return early as there are no centroids or tracking info
            # to update
            return self

        # initialize an array of input centroids for the current frame
        inputCentroids = np.array((centroids), dtype='int')

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
            # otherwise, are are currently tracking objects so we need to
            # try to match the input centroids to existing object
            # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            objectCentroids = np.array([*objectCentroids]).reshape((-1, 2))
            pairDistance = dist.cdist(objectCentroids, inputCentroids)

            # find the smallest value in each row and then sort the row
            # indexes based on their minimum values so the row
            # with the smallest value is at the *front* of the index
            # list
            rows = pairDistance.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = pairDistance.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = np.array(inputCentroids[col])
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, pairDistance.shape[0])).difference(usedRows)
            unusedCols = set(range(0, pairDistance.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if pairDistance.shape[0] >= pairDistance.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                       self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

            self.updateVehicleCounter()

            # return the class object
            return self

    def deregister(self, objectID):
        """To deregister an object ID we delete the object ID from
        all of our respective dictionaries
        """

        del self.objects[objectID]
        del self.objectFramePick[objectID]
        del self.disappeared[objectID]

    def updateVehicleCounter(self):
        """Updates the vehicle counter with the last inserted
        object index in the objects instance
        """

        # If the difference between the new tracking objects and the previous
        # number of objects is greater than 3, add the maximum of 3 objects
        # in the counter. We have made the assumption, at each frame only
        # 3 new objects pass through the window at a given frame. This states
        # from the 3 lanes at each current
        if abs(self.vehicleCounter - self.nextObjectID + 1) > 3:
            self.vehicleCounter += 3
        else:
            self.vehicleCounter = self.nextObjectID + 1

    def getVehicleCounter(self):
        """Returns the number of annotated vehicles
        """
        return self.vehicleCounter


