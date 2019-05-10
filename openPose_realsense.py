import cv2
import time
import numpy as np
import tensorflow as tf
import pyrealsense2 as rs

inWidth = 368
inHeight = 368
threshold = 0.1

nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


# input_source = "samples/05_0000_RGB.mp4"
# cap = cv2.VideoCapture(input_source)
# #cap = cv2.VideoCapture(0)
# hasFrame, frame = cap.read()

#Ini camera
SIZE = (640, 480)
FPS = 30
MIN_DEPTH_R = 170
MAX_DEPTH_R = 380
pipelineR = rs.pipeline()
configR = rs.config()
configR.enable_device('819112070557')
configR.enable_stream(rs.stream.depth, SIZE[0], int(0.75*SIZE[1]), rs.format.z16, FPS)
configR.enable_stream(rs.stream.color, SIZE[0], SIZE[1], rs.format.bgr8, FPS)
# Start streaming
profileR = pipelineR.start(configR)
alignR = rs.align(rs.stream.color)
depth_sensor = profileR.get_device().first_depth_sensor()
depth_scale_R = depth_sensor.get_depth_scale()
min_depthR_data = int(MIN_DEPTH_R*0.01 / depth_scale_R)
max_depthR_data = int(MAX_DEPTH_R*0.01 / depth_scale_R)
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale_R








def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph
 
graph = load_graph('model/pose.pb')

net_input = graph.get_tensor_by_name('prefix/image:0')
net_output = graph.get_tensor_by_name('prefix/concat_stage7:0')

sess = tf.Session(graph=graph)

back_flag = False

while cv2.waitKey(1) < 0:
    t = time.time()
    #hasFrame, frame = cap.read()
    #frameCopy = np.copy(frame)

    frames = pipelineR.wait_for_frames()
    aligned_frames = alignR.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()    
    if not aligned_depth_frame or not color_frame:
        hasFrame = False
    else:
        hasFrame = True

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())
    frameCopy = np.copy(frame)


    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    inpBlob = np.swapaxes(inpBlob,1,3)
    inpBlob = np.swapaxes(inpBlob,1,2)
    output = sess.run(net_output, feed_dict={ net_input: inpBlob })
    output = np.swapaxes(output,1,3)
    output = np.swapaxes(output,2,3)

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    #frame[:] = 0

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)
