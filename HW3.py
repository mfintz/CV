import cv2, numpy as np
import random
import sys
import math

#
# Globals for the DETECTION configuration
#
GAUSSIAN_BLUR_SIZE = 13
THRESHOLD_AFTER_BULR = 120
MIN_CONTOUR_AREA = 100
CANDIDATE_MAX_DISTANCE = 40
TTL_NO_UPDATE = 2

#
# Globals for the PROGRAM configuration
#
RUN_FRAME_BY_FRAME = True
SHOW_KALMAN_PREDICTIONS = False
START_DETECTION_Y_COORDINATE = 150
SHOW_SUBTRACTION_MASK = False
RED_LINE_POSITION_PERCENTAGE = 0.8

#
# Globals for the car detection
#
TrackedCars = []
NextCarID = 0
Step = 0

#
# A structure to keep all info for a single tracked car
# Each tracked car has its own kalman predictor to be used when the system was unable to find a suitable position
#
class TrackedCar:
    x = 0
    y = 0
    width = 0
    height = 0
    color = (0, 0, 0)
    id = 0
    stepUpdated = 0
    kalmanFilter = cv2.KalmanFilter(4,2)
    predictedPosition = False
    passedRedLine = False

def GetRandomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def math_calc_dist(p1,p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) +
                     math.pow((p2[1] - p1[1]), 2))

def ExtractCarsFromFGMask(fgmask):
    global GAUSSIAN_BLUR_SIZE, THRESHOLD_AFTER_BULR, MIN_CONTOUR_AREA, START_DETECTION_Y_COORDINATE
    detectedCarsRects = []

    #
    # Process the foreground binary image before finding contours
    #
    fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)[1]
    fgmask = cv2.GaussianBlur(fgmask, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    fgmask = cv2.threshold(fgmask, THRESHOLD_AFTER_BULR, 255, cv2.THRESH_BINARY)[1]
    _, contours, _  = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    #
    # loop over the contours
    #
    for c in contours:
        #
        # if the contour is too small, ignore it
        #
        if cv2.contourArea(c) < MIN_CONTOUR_AREA :
           continue

        #
        # If the car above the detection line - ignore it
        #
        boundingRect = cv2.boundingRect(c)
        if boundingRect[1] < START_DETECTION_Y_COORDINATE:
           continue

        #
        # Here we add the bounding rect as a detected car
        #
        detectedCarsRects.append(boundingRect)
    return detectedCarsRects

def AddRectsToTrackedCars(rects):
    global NextCarID, Step, TrackedCars

    if not rects:
        return

    for rect in rects:
        trackedCar = TrackedCar()
        trackedCar.id = NextCarID
        NextCarID = NextCarID + 1
        trackedCar.color = GetRandomColor()
        trackedCar.x = rect[0]
        trackedCar.y = rect[1]
        trackedCar.width = rect[2]
        trackedCar.height = rect[3]
        trackedCar.stepUpdated = Step
        trackedCar.kalmanFilter = cv2.KalmanFilter(4,2)
        trackedCar.kalmanFilter.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        trackedCar.kalmanFilter.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        trackedCar.kalmanFilter.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

        #
        # Initialize Kalman
        #
        for i in range(0):
            trackedCar.kalmanFilter.correct(np.array([np.float32(trackedCar.x),np.float32(trackedCar.y + i)]))
            trackedCar.kalmanFilter.predict()

        #
        # Add to the traced cars list
        #
        TrackedCars.append(trackedCar);

def ProcessDetectedCars(rects):
    global CANDIDATE_MAX_DISTANCE

    if not rects:
        return

    for trackedCar in TrackedCars:
        candidateIndex = -1
        candidateDistance = 1000000

        #
        # Find the closest rect to the given car
        #
        kalmanPredict = trackedCar.kalmanFilter.predict()
        print ("Original/Kalman coordinates for " + str(trackedCar.id) + ": (" + str(trackedCar.x) + "," + str(trackedCar.y) + ")/(" + str(kalmanPredict[0]) + "," + str(kalmanPredict[1]) + ")")
        
        #
        # A simple next-step detection algorythm - find the closest detected object (a candidate)
        # We assume that cars always move in one direction - down the frame
        #
        i = 0
        for rect in rects:
            if (rect[1] > trackedCar.y - 1):
                distance = math_calc_dist((rect[0], rect[1]), (trackedCar.x, trackedCar.y))
                if (distance < candidateDistance):
                    candidateDistance = distance
                    candidateIndex = i
            i = i + 1

        #
        # If a candidate wasn't found
        #
        if candidateIndex == -1 or candidateDistance > CANDIDATE_MAX_DISTANCE :
            trackedCar.x = kalmanPredict[0][0]
            trackedCar.y = kalmanPredict[1][0]
            trackedCar.predictedPosition = True
        else:
            #
            # Update the coordinate in the tracked car and delete the candidate from the list
            #
            rectCandidate = rects[candidateIndex];
            trackedCar.x = rectCandidate[0]
            trackedCar.y = rectCandidate[1]
            trackedCar.predictedPosition = False
            trackedCar.width = rectCandidate[2]
            trackedCar.height = rectCandidate[3]
            trackedCar.stepUpdated = Step
            del rects[candidateIndex]

    #
    # Call the correction in any case
    #
    trackedCar.kalmanFilter.correct(np.array([[np.float32(trackedCar.x)],[np.float32(trackedCar.y)]]))

    #
    # Now, the rest of the detected rects ARE new cars and they should be added to the trackedCars list
    #
    AddRectsToTrackedCars(rects)

def TrackVideo(filename):
    global Step, SHOW_SUBTRACTION_MASK, START_DETECTION_Y_COORDINATE, TTL_NO_UPDATE, SHOW_KALMAN_PREDICTIONS, RUN_FRAME_BY_FRAME, TrackedCars, RED_LINE_POSITION_PERCENTAGE

    cap = cv2.VideoCapture(filename)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    #
    # Read the first frame to get its size
    #
    ret, frame = cap.read()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    RedLineYPosition = int(RED_LINE_POSITION_PERCENTAGE * frameHeight)

    #
    # Run first frames to calculate the initial background
    #
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        Rects = ExtractCarsFromFGMask(fgmask)
        if not not Rects:
            break

    #
    # After the first cars are detected - initialize the variables and register these first cars as tracked cars
    #
    passedRedLineCount = 0
    AddRectsToTrackedCars(Rects)

    #
    # At this point fgmask contains the initial tracked frame
    # We extract cars to track and will continue tracking them (and new ones) in a loop over the rest of frames
    #
    framesCount = 0
    while(1):
        ret, frame = cap.read()
        if ret == False:
            break

        fgmask = fgbg.apply(frame)
        framesCount = framesCount + 1

        if SHOW_SUBTRACTION_MASK == True:
            copymask = fgmask.copy()

        #
        # Extract cars
        #    
        rects = ExtractCarsFromFGMask(fgmask)
        if not rects:
            continue

        Step = Step + 1
        ProcessDetectedCars(rects)

        #
        # Tracked cars - draw the rectangles and delete old cars to improve performance
        #
        trackedIndex = -1
        for trackedCar in TrackedCars:
            #
            # If no tracking is available for the given car for more than TTL_NO_UPDATE steps
            # this may mean: 1. The car passed below the frame
            #                2. The car was a detection mistake and now joined another car
            #                3. The noise was detected and thought of as a car
            # These are garbage cars and should be removed
            #
            trackedIndex = trackedIndex + 1
            if trackedCar.stepUpdated < Step - TTL_NO_UPDATE:
                del TrackedCars[trackedIndex]
                continue

            #
            # Immediately remove from the list cars that moved beyond the end of the frame to IMPROVE PERFORMANCE
            # This step may be omitted and then these cars will be deleted as a garbage cars in the next several iterations
            #
            if trackedCar.y > frameHeight:
                del TrackedCars[trackedIndex]
                continue

            #
            # Now we will draw the rectangles on any tracked car that we detected in the last frame.
            # There are cars that tracking information was not available from this frame - in this case it's position is predicted by Kalman
            # But still - we don't draw them on the frame although don't delete them immediately as they may be redetected in the next frames
            #
            if trackedCar.stepUpdated == Step:
                (x, y, w, h) = (int(trackedCar.x), int(trackedCar.y), trackedCar.width, trackedCar.height)
                cv2.rectangle(frame, (x, y), (x + w, y + h), trackedCar.color, 2)
                cv2.putText(frame, str(trackedCar.id),
                        (trackedCar.x, trackedCar.y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                #
                # A bonus track - count red line crossing cars
                #
                if (y > RedLineYPosition) and trackedCar.passedRedLine == False:
                    trackedCar.passedRedLine = True
                    passedRedLineCount = passedRedLineCount + 1

                #
                # For debug purposes we can show the subtraction mask, this may be configured by the global parameter SHOW_SUBTRACTION_MASK
                #
                if SHOW_SUBTRACTION_MASK == True:
                    cv2.rectangle(copymask, (x, y), (x + w, y + h), trackedCar.color, 2)
                    cv2.putText(copymask, str(trackedCar.id),
                        (trackedCar.x, trackedCar.y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            else:
                #
                # For debug purposes we can show Kalman predictions in black. This is configured by the global parameter SHOW_KALMAN_PREDICTIONS
                #
                if trackedCar.predictedPosition == True and SHOW_KALMAN_PREDICTIONS == True:
                    (x, y, w, h) = (int(trackedCar.x), int(trackedCar.y), trackedCar.width, trackedCar.height)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    cv2.putText(frame, str(trackedCar.id),
                        (trackedCar.x, trackedCar.y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
        #
        # Put Red and Green lines on the frame
        #
        cv2.line(frame, (0, START_DETECTION_Y_COORDINATE), (frameWidth, START_DETECTION_Y_COORDINATE), (0, 255, 0), 1)
        cv2.line(frame, (0, RedLineYPosition), (frameWidth, RedLineYPosition), (0, 0, 255), 1)

        #
        # Put texts on the frame
        #
        cv2.putText(frame, "Frame: " + str(framesCount),
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 1)
        cv2.putText(frame, "Passed the red line: " + str(passedRedLineCount),
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 1)
        
        #
        # Show the frame, the mask, etc
        #
        if SHOW_SUBTRACTION_MASK == True:
            cv2.imshow('frame_MASK', copymask)
        cv2.imshow(filename, frame)

        #
        # Exit criteria and step-by-step run
        #
        if RUN_FRAME_BY_FRAME == True:
            cv2.waitKey(0)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()

#
# Main
#
TrackVideo('Highway1.avi')
TrackVideo('Highway3.avi')
TrackVideo('Highway4.avi')
