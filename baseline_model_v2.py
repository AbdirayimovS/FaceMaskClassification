import cv2 as cv
import mediapipe as mp
import math
import numpy as np
import tensorflow as tf

face_dt = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
color = (0,255,0)
thickness = 2
mask_status=None
cap = cv.VideoCapture(0)

rect_start_point, rect_end_point = (0,0), (0,0)
facedet = face_dt.FaceDetection(model_selection=0,
                                min_detection_confidence=0.5)
saved_model = tf.keras.models.load_model("saved_models/my_model_kaggle_course_v3")

def normalized_to_pixel_coor( normalized_x, normalized_y, im_width, im_height):
    x_px = min(math.floor(normalized_x * im_width), im_width-1)
    y_px = min(math.floor(normalized_y * im_height), im_height-1)
    return x_px,y_px

while cap.isOpened():
    success, frame = cap.read()
    im_height, im_width,_ = frame.shape
    frame.flags.writeable = False
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = facedet.process(image)
    frame.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.detections:
        for detection in results.detections:
            bd = detection.location_data.relative_bounding_box
            rect_start_point = normalized_to_pixel_coor(bd.xmin, bd.ymin, im_width, im_height)
            rect_end_point = normalized_to_pixel_coor(bd.xmin+bd.width, bd.ymin+bd.height, im_width, im_height)
            frame = cv.rectangle(image, rect_start_point,rect_end_point,color, thickness)

            ####prediction_part
            try:
                img = frame[rect_start_point[1]:rect_end_point[1],
                           rect_start_point[0]:rect_end_point[0]]
                img = cv.resize(img, (128, 128))
                img = np.array(img)
                final_face = tf.expand_dims(img, axis=0)
                ypred = saved_model.predict(final_face)
                if int(ypred[0]) ==0:
                    mask_status="Without Mask"
                else:
                    mask_status="With Mask"
                print(ypred)
                cv.putText(frame, mask_status, (50, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, color, thickness,
                       cv.LINE_AA)
            except Exception as e:
                pass


    cv.imshow("Face Detection:", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break



cap.release()
cv.destroyAllWindows()



