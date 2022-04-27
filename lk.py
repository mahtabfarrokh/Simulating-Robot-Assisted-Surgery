import os
import sys
import cv2
import numpy as np
import math
import time


def readTrackingData(filename):
    if not os.path.isfile(filename):
        print("Tracking data file not found:\n ", filename)
        sys.exit()

    data_file = open(filename, 'r')
    lines = data_file.readlines()
    no_of_lines = len(lines)
    data_array = np.zeros((no_of_lines, 8))
    line_id = 0
    for line in lines:
        words = line.split()[1:]
        if len(words) != 8:
            msg = "Invalid formatting on line %d" % line_id + " in file %s" % filename + ":\n%s" % line
            raise SyntaxError(msg)
        coordinates = []
        for word in words:
            coordinates.append(float(word))
        data_array[line_id, :] = coordinates
        line_id += 1
    data_file.close()
    return data_array


def writeCorners(file_id, corners):
    # write the given corners to the file
    corner_str = ''
    for i in range(4):
        corner_str = corner_str + '{:5.2f}\t{:5.2f}\t'.format(corners[0, i], corners[1, i])
    file_id.write(corner_str + '\n')


def drawRegion(img, corners, color, thickness=1):
    # draw the bounding box specified by the given corners
    for i in range(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)


def initTracker(img, corners):
    # initialize your tracker with the first frame from the sequence and
    # the corresponding corners from the ground truth
    # this function does not return anything
    global old_frame
    global p0
    old_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p0 = corners.T.astype(np.float32)
    pass


def updateTracker(img):
    # update your tracker with the current image and return the current corners
    # at present it simply returns the actual corners with an offset so that
    # a valid value is returned for the code to run without errors
    # this is only for demonstration purpose and your code must NOT use actual corners in any way
    global old_frame
    global p0
    frame_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (32,32),
                  maxLevel = 8,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 9, 0.02555))
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_img, p0, None, **lk_params)
    old_frame = frame_img.copy()
    p0 = p1.copy()

    return p1.T




seq_name = 'box'

src_fname = seq_name + '/frame%05d.jpg'
ground_truth_fname = seq_name + '.txt'
result_fname = seq_name + '_res.txt'

result_file = open(result_fname, 'w')

cap = cv2.VideoCapture()
if not cap.open(src_fname):
    print('The video file ', src_fname, ' could not be opened')
    sys.exit()


# read the ground truth
#ground_truth = readTrackingData(ground_truth_fname)



no_of_frames = ground_truth.shape[0]

ret, init_img = cap.read()
if not ret:
    print("Initial frame could not be read")
    sys.exit(0)

# extract the true corners in the first frame and place them into a 2x4 array
init_corners = [ground_truth[0, 0:2].tolist(),
                ground_truth[0, 2:4].tolist(),
                ground_truth[0, 4:6].tolist(),
                ground_truth[0, 6:8].tolist()]
init_corners = np.array(init_corners).T
# write the initial corners to the result file
writeCorners(result_file, init_corners)

# initialize tracker with the first frame and the initial corners
initTracker(init_img, init_corners)


for frame_id in range(1, no_of_frames):
    ret, src_img = cap.read()
    if not ret:
        print("Frame ", frame_id, " could not be read")
        break


    tracker_corners = updateTracker(src_img)

    # write the current tracker location to the result text file
    writeCorners(result_file, tracker_corners)


result_file.close()

