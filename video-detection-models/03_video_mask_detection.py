# Source code was: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
# Inspired by presentation from Claire Hester on her model for face mask detection

# Bringing some mates
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os



# The below deploys our model, tracks the face, and keeps a bounding box on faces
def detect_and_predict_mask(frame, face_detector, model):
    # grab the frame dimensions and construct a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    # initialize list of faces, locations, and predictions
    faces = []
    locs = []
    preds = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract confidence 
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box 
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # ensure the bounding boxes fall within the frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # extract the face ROI and preprocess
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (256, 256))
            face = img_to_array(face)
            face = preprocess_input(face)
            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    # Make predictions when faces are detected
    if len(faces) > 0:
        
        faces = np.array(faces, dtype="float32")
        preds = model.predict(faces, batch_size=32)
    # return a tuple of the face locations and their corresponding predictions
    return (locs, preds)


print("Loading Face Mask Detection Model")
prototxtPath = os.path.sep.join(['./opencv_face_detector/', "deploy.prototxt"])
weightsPath = os.path.sep.join(['./opencv_face_detector/', "res10_300x300_ssd_iter_140000.caffemodel"])
face_detector = cv2.dnn.readNet(prototxtPath, weightsPath)
# the below loads our model
print("Loading Face Mask Detection Model")
model = load_model('face_model_2.h5')
# Start the video stream with the below
print("Video Stream Is Beginning")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# log start time and start frame count
start_time = time.time()
frame_count = 0

# loop over the frames from the video stream
while True:
    # track frame count
    frame_count +=1
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
    # run function to make predictions
    (locs, preds) = detect_and_predict_mask(frame, face_detector, model)



# loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        Unmasked, Masked, Incorrectly_Masked = pred
        # determine the class label and color we'll use to draw the bounding box and text
        # Just a note, the colors below are BGR, if you want to change the colors of the bounding boxes
        if max([Unmasked, Masked, Incorrectly_Masked]) == Masked:
            label = 'Masked'
            color = (0, 255, 0)
        elif max([Unmasked, Masked, Incorrectly_Masked]) == Unmasked:
            label = 'Unmasked'
            color = (0, 0, 255)
        else:
            label = 'Incorrectly_Masked'
            color = (0, 255, 255)
        # Including probability in the labeling
        label = "{}: {:.2f}%".format(label, max([Unmasked, Masked, Incorrectly_Masked]) * 100)
        # This displays the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


    # This displays the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # Break the video looping by pressing the 'q' key
    if key == ord("q"):
        break

# Calculate the time lapsed and display it in the command line
time_lapsed = time.time() - start_time
fps = frame_count / time_lapsed
print(f'Total time: {time_lapsed} Frames per second: {fps}')
    
# Clean up
cv2.destroyAllWindows()
vs.stop()
