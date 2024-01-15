import numpy as np
import cv2
import sys
from numpy import ones,vstack
from numpy.linalg import lstsq
import  math
import tkinter as tk
from tkinter import filedialog

#
# Global variables
#
DEBUG_OUTPUT = True
DEBUG_DRAW_VERBOSE = True
IgnoreVerticalParallelLines = False
SECTION_HIGHLIGHT_COLOR = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
drawLine = False
segmentStart_X = 0
segmentStart_Y = 0
currentSection = 0

undo_available = False
segments = [[], [], []]
lines = [[], [], []]
vanishing_points = []

REFERENCE_HIGHLIGHT_COLOR = [255, 255, 255]
MEASURE_HIGHLIGHT_COLOR = [255, 255, 0]
referenceStart_X = 0
referenceStart_Y = 0
referenceStop_X = 0
referenceStop_Y = 0
referenceBottom = []
referenceTop = []
referenceHeight = 0

workingImage = None
accumulatedImage = None
undo_img = None

measureStart_X = 0
measureStart_Y = 0
measureStop_X = 0
measureStop_Y = 0

################# MOUSE CALLBACKS ##############
#
# We extensively use mouse work so we have several diffent mouse callback for each part of the HW
#

#
# A mouse callback to serve user's measurements requests
#
def mouse_CollectMeasurementRequests(event, x, y, flags, param):
    global workingImage, accumulatedImage, drawLine, REFERENCE_HIGHLIGHT_COLOR, measureStart_X, measureStart_Y, measureStop_X, measureStop_Y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawLine = True
        measureStart_X = x
        measureStart_Y = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawLine == True:
            workingImage = accumulatedImage.copy()
            cv2.line(workingImage, (measureStart_X, measureStart_Y), (x, y), MEASURE_HIGHLIGHT_COLOR, 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawLine = False
        accumulatedImage = workingImage.copy()
        measureStop_X = x
        measureStop_Y = y
        height = CalculateHeight(measureStart_X, measureStart_Y, measureStop_X, measureStop_Y)
        cv2.putText(accumulatedImage, str(height),
                        (measureStart_X, int((measureStart_Y + measureStop_Y)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, MEASURE_HIGHLIGHT_COLOR, 1)
        workingImage = accumulatedImage.copy()

#
# A mouse callback to collect the reference VERTICAL measurement 
#
def mouse_CollectReference(event, x, y, flags, param):
    global workingImage, accumulatedImage, drawLine, REFERENCE_HIGHLIGHT_COLOR, referenceStop_X, referenceStop_Y, referenceStart_X, referenceStart_Y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawLine = True
        referenceStart_X = x
        referenceStart_Y = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawLine == True:
            workingImage = accumulatedImage.copy()
            cv2.line(workingImage, (referenceStart_X, referenceStart_Y), (x, y), REFERENCE_HIGHLIGHT_COLOR, 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawLine = False
        accumulatedImage = workingImage.copy()
        referenceStop_X = x
        referenceStop_Y = y
    return 

#
# A mouse callback used to collect segments, supports undo for the last segment attempt
#
def mouse_collectSegments(event, x, y, flags, param):
    global workingImage, accumulatedImage, undo_img, drawLine, undo_available, segmentStart_X, segmentStart_Y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawLine = True
        segmentStart_X = x
        segmentStart_Y = y
        undo_img = accumulatedImage.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawLine == True:
            print (currentSection)
            workingImage = accumulatedImage.copy()
            cv2.line(workingImage, (segmentStart_X, segmentStart_Y), (x, y), SECTION_HIGHLIGHT_COLOR[currentSection], 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawLine = False
        undo_available = True
        accumulatedImage = workingImage.copy()

        #
        # If this is a simple one-click - it should be ignored
        #
        if not(segmentStart_X == x and segmentStart_Y == y):
            segments[currentSection].append([[segmentStart_X, segmentStart_Y], [x, y]])
            print(segments[currentSection])

    return
#
# An empty mouse callback
#
def EmptyCallback(event, x, y, flags, param):
    return


#################### LOGIC ###############
#
#
#

#
#  A simple method to pick a file
#
def selectfile():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

#
# This method initiates the image and collects segments by attaching the corresponding callback
#
def CollectSegmentsData():
    global workingImage, accumulatedImage, undo_img, undo_available, currentSection

    #
    # Create a named window and display the image
    #
    workingImage = cv2.imread(selectfile())
    accumulatedImage = workingImage.copy()
    undo_img = workingImage.copy()
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', mouse_collectSegments)
    cv2.moveWindow('input', workingImage.shape[1]+10, 90)
    
    print("By pressing 1, 2 or 3  select the axis X, Y and Z correspondingly\n")
    print("After axis selection drag segments. You can select axis several times\n")
    print("Press Enter when finished with segments marking\n")
    while(1):
        cv2.imshow('input', workingImage)
        k = 0xFF & cv2.waitKey(13)

        if k == 13:         # Enter - stop collecting data
            print("Segments collected\n")
            break
        elif k == ord('1'):
            print("Add an axis X segment\n")
            currentSection = 0
            undo_available = False
        elif k == ord('2'):
            print("Add an axis Y segment\n")
            currentSection = 1
            undo_available = False
        elif k == ord('3'):
            print("Add an axis X segment\n")
            currentSection = 2
            undo_available = False
        #
        # Some nice feature which was supposed by other students's team
        # We included it here for the convenience of the checker ))
        #
        elif k == ord('u'):
            if undo_available:
                print("Undo performed")
                workingImage = undo_img.copy()
                accumulatedImage = workingImage.copy()
                segments[currentSection].pop()
                undo_available = False
            else:
                print("Undo unavailable")

    #
    # Disconnect the callback meanwhile
    #
    cv2.setMouseCallback('input', EmptyCallback)
    return

#
# This method calculates the normal line equation parameters from the segments (points).
# Ax + By = C
#
def GalculateLinesFromSegments():
    global segments, lines
    for i in range(0, 3):
        sectionSegments = segments[i]
        for segment in sectionSegments:
            x1 = segment[0][0]
            y1 = segment[0][1]
            x2 = segment[1][0]
            y2 = segment[1][1]
            points = [(x1,y1),(x2,y2)]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords,ones(len(x_coords))]).T

            #
            # We use the LSQ approach to solve the equations
            #
            m, c = lstsq(A, y_coords)[0] # y = mx + c
            newLine = [m, -1, - c]
            lines[i].append(newLine)

            if DEBUG_OUTPUT == True:
                print("Added line: ")
                print(newLine)
                print("\n")
    return

#
# This method calculates the vanishing points using the LSQ to find the best intersection point
#
def CalculateVanishingPoints():
    global lines, vanishing_points, accumulatedImage, workingImage, DEBUG_OUTPUT, SECTION_HIGHLIGHT_COLOR

    #
    # And now for each axis
    #
    for i in range(0, 3):
        #
        # We build the matrix from the line A and B coefficients
        #
        A = []
        B = []
        C = []
        sectionLines = lines[i]
        for line in sectionLines:

            #
            # Fill the lists with the coefficients
            #
            A_coeff = line[0]
            B_coeff = line[1]
            C_coeff = line[2]
            A.append(A_coeff)
            B.append(B_coeff)
            C.append(C_coeff)

        #
        # Building the linear algebra matrix form A and B coefficients
        #
        matrixA = np.array([[A[j], B[j]] for j in range(len(A))])

        #
        # Running the LSQ with the C vector to find t he best intersection point
        #
        solution = np.linalg.lstsq(matrixA, C)[0]
        vanishing_points.append(solution)

        #
        # A special case - parallel vertical lines
        # We miss here the case when the real vertical lines meet at the Y-centre of the image
        # In future we can ask user here about it
        #
        if i == 3 and abs(solution[1] - 0) < 0.1:
            IgnoreVerticalParallelLines = True

        if DEBUG_OUTPUT == True:
            print("For Axis ")
            print(i)
            print(" the vanishing point is")
            print(solution)
            print("\n")

            #
            # Here we will draw the lines which go the vanishing points (continue segments)
            #
            sectionSegments = segments[i]
            for segment in sectionSegments:
                x1 = segment[0][0]
                y1 = segment[0][1]
                x2 = segment[1][0]
                y2 = segment[1][1]
                cv2.line(accumulatedImage, (x1, y1), (int(solution[0]), int(solution[1])), SECTION_HIGHLIGHT_COLOR[i], 1)
                cv2.line(accumulatedImage, (x2, y2), (int(solution[0]), int(solution[1])), SECTION_HIGHLIGHT_COLOR[i], 1)
            workingImage = accumulatedImage.copy()

    #
    # Here we will draw the bold vanishing line (the horizon)
    #
    if DEBUG_OUTPUT == True:
        cv2.line(accumulatedImage, (int(vanishing_points[0][0]), int(vanishing_points[0][1])), (int(vanishing_points[1][0]), int(vanishing_points[1][1])), [255, 255, 255], 3)
        workingImage = accumulatedImage.copy()
    return

#
# This method collects vertical reference from user. It enables the corresponding mouse callback
#
def CollectReference():
    global referenceHeight, accumulatedImage, workingImage, referenceStart_X, referenceStart_Y, referenceBottom, referenceTop
    cv2.setMouseCallback('input', mouse_CollectReference)
    print("Drag the reference and press Enter. After this enter the reference's height in the console window (int expected)\n")
    print("Only the last attempt counts\n")

    while(1):
        cv2.imshow('input', workingImage)
        k = 0xFF & cv2.waitKey(13)
        if k == 13:
            print("Reference selected\n")
            break

    #
    # Determining the Top and the Bottom
    #
    if(referenceStart_Y < referenceStop_Y):
        referenceBottom.append(referenceStart_X)
        referenceBottom.append(referenceStart_Y)
        referenceTop.append(referenceStop_X)
        referenceTop.append(referenceStop_Y)
    else:
        referenceBottom.append(referenceStop_X)
        referenceBottom.append(referenceStop_Y)
        referenceTop.append(referenceStart_X)
        referenceTop.append(referenceStart_Y)

    #
    # Input from the user
    #
    referenceHeight = int(input("Enter the reference's height: "))

    #
    # Write the height on the image
    #
    cv2.putText(accumulatedImage, str(referenceHeight),
                        (referenceStart_X, referenceStart_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, REFERENCE_HIGHLIGHT_COLOR, 1)
    cv2.imshow('input', accumulatedImage)
    workingImage = accumulatedImage.copy()

    return

#
# A simple math routine to calculate the distance
#
def math_calc_dist(p1,p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) +
                     math.pow((p2[1] - p1[1]), 2))

def math_calc_dist_homogenous(p1,p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) +
                     math.pow((p2[1] - p1[1]), 2) +
                     math.pow((p2[2] - p1[2]), 2))
#
# A simple routine to get back from the homogenous coordinates
#
def NormalizeHomogenous(v):
    result = []
    z = v[2]
    result.append(v[0] / z)
    result.append(v[1]/z)
    return result

#
# This routine calculates the height - here is the main logic.
# It should run from the corresponding mouse callback
#
def CalculateHeight(x1, y1, x2, y2):
    global referenceBottom, vanishing_points, referenceHeight, referenceTop, workingImage, accumulatedImage, DEBUG_OUTPUT

    #
    # We will use the same notation as in the lecture slides, e.g. vx, vy, vz, t, t0, b, b0, etc
    #
    b = referenceBottom
    r = referenceTop
    vx = vanishing_points[0].tolist()
    vy = vanishing_points[1].tolist()
    vz = vanishing_points[2].tolist()
    b0 = []
    if y1 < y2:
        b0 = [x1, y1]
        t0 = [x2, y2]
    else:
        b0 = [x2, y2]
        t0 = [x1, y1]

    #
    # Transfer some points into the Homogenous coordinates
    #
    t0_H = t0.copy()
    t0_H.append(1)
    r_H = r.copy()
    r_H.append(1)
    b_H = b.copy()
    b_H.append(1)
    b0_H = b0.copy()
    b0_H.append(1)
    vx_H = vx.copy()
    vx_H.append(1)
    vy_H = vy.copy()
    vy_H.append(1)
    vz_H = vz.copy()
    vz_H.append(1)

    #
    # Calculating the cross products (possible only for 3D vectors
    #
    v_H = np.cross(np.cross(b_H, b0_H), np.cross(vx_H, vy_H))
    t_H = np.cross(np.cross(v_H, t0_H), np.cross(r_H, b_H))

    #
    # Going back to normal 2D coordinates
    #
    v = NormalizeHomogenous(v_H)
    t = NormalizeHomogenous(t_H)

    #
    # Here we will draw the projection lines
    #
    if DEBUG_OUTPUT == True:
        cv2.line(accumulatedImage, (int(v[0]), int(v[1])), (int(t[0]), int(t[1])), [255,255,255], 1)
        cv2.line(accumulatedImage, (int(v[0]), int(v[1])), (int(b[0]), int(b[1])), [255,255,255], 1)

        if DEBUG_DRAW_VERBOSE == True:
            cv2.putText(accumulatedImage, "b",
                            (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_HIGHLIGHT_COLOR, 1)
            cv2.putText(accumulatedImage, "b0",
                            (int(b0[0]), int(b0[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_HIGHLIGHT_COLOR, 1)
            cv2.putText(accumulatedImage, "t",
                            (int(t[0]), int(t[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_HIGHLIGHT_COLOR, 1)
            cv2.putText(accumulatedImage, "t0",
                            (int(t0[0]), int(t0[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_HIGHLIGHT_COLOR, 1)
            cv2.putText(accumulatedImage, "r",
                            (int(r[0]), int(r[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_HIGHLIGHT_COLOR, 1)
            cv2.putText(accumulatedImage, "v",
                            (int(v[0]), int(v[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, REFERENCE_HIGHLIGHT_COLOR, 1)
        workingImage = accumulatedImage.copy()

    #
    # Performing the famous cross-ratio calculations
    #
    H = referenceHeight * math_calc_dist(t, b)/ math_calc_dist(r, b)
 
    #
    # Add the vertical correction in case that we have a real vz
    #
    if IgnoreVerticalParallelLines == False:
        H = H * math_calc_dist(vz, r) / math_calc_dist(vz, t)
    return int(H)

#
# This routine collects user's measurement request and calculates the height.
# It employs the corresponding callback and waits to the Escape key
#
def ServeUserMeasurementRequests():
    global workingImage
    cv2.setMouseCallback('input', mouse_CollectMeasurementRequests)

    while(1):
        cv2.imshow('input', workingImage)
        k = 0xFF & cv2.waitKey(13)
        if k == 27:
            break
    return

#
# Main
#
CollectSegmentsData()
GalculateLinesFromSegments()
CalculateVanishingPoints()
CollectReference()
ServeUserMeasurementRequests()
