### The following hand-detection code is by cvzone(https://www.computervision.zone/) the video it is taken from: https://www.youtube.com/watch?v=RQ-2JWzNc6k
import cv2
import math
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import random

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info

def hand_proc(img):
    hands, img = detector.findHands(img)

    data = []
    if hands:
        # Hand 1
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 Landmark points
        for lm in lmList:
            data.extend([lm[0], h - lm[1], lm[2]])

def guitar_proc(img):
    results = model_guitar_detect(img)[0]
    cropped_img = None
    angle = None

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if score > threshold_guitar_detect:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(img, results.names[int(class_id)], (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            break 

    return cropped_img, angle

def canny_hough_lines(img, canny_lower, canny_upper, threshold=220, angle_threshold=45, line_length=50):
    gray_scale = img.copy()
    gray_scale = cv2.cvtColor(gray_scale, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_scale, canny_lower, canny_upper, None, 3)
    
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + line_length * (-b)), int(y0 + line_length * (a)))
            pt2 = (int(x0 - line_length * (-b)), int(y0 - line_length * (a)))
            cv2.line(cdst, pt1, pt2, (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)), 1, cv2.LINE_AA)
            
    return cdst

def canny_hough_line_prob(img, canny_lower, canny_upper, threshold=80, angle_threshold=45, tolerance=10):
    gray_scale = img.copy()
    gray_scale = cv2.cvtColor(gray_scale, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_scale, canny_lower, canny_upper, None, 3)
    
    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold)
    slopes_lines = {}

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cdst, (x1, y1), (x2, y2), (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)), 1, cv2.LINE_AA)
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = np.inf
            
            if slope in slopes_lines:
                slopes_lines[slope].append(line)
            else:
                slopes_lines[slope] = [line]
    
    for key in slopes_lines:
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        for line in slopes_lines[key]:
            x1, y1, x2, y2 = line[0]
            min_x = min(min_x, x1, x2)
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)

        cv2.line(cdst, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1, cv2.LINE_AA)
    
    return cdst
    
def canny(img):
    gray_scale = img.copy()
    gray_scale = cv2.cvtColor(gray_scale, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_scale, canny_lower, canny_upper, None, 3)
    img_return = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return img_return

def pred_lines(image, interpreter, input_details, output_details, input_shape=[512, 512], score_thr=0.10, dist_thr=20.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA), np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], batch_image)
    interpreter.invoke()

    pts = interpreter.get_tensor(output_details[0]['index'])[0]
    pts_score = interpreter.get_tensor(output_details[1]['index'])[0]
    vmap = interpreter.get_tensor(output_details[2]['index'])[0]

    start = vmap[:,:,:2]
    end = vmap[:,:,2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
    
    lines = 2 * np.array(segments_list) # 256 > 512
    lines[:,0] = lines[:,0] * w_ratio
    lines[:,1] = lines[:,1] * h_ratio
    lines[:,2] = lines[:,2] * w_ratio
    lines[:,3] = lines[:,3] * h_ratio

    return lines

model_m_lsd = "M-LSD_320_tiny_fp16.tflite"
interpreter_m_lsd = tf.lite.Interpreter(model_path=model_m_lsd)

interpreter_m_lsd.allocate_tensors()
input_details_m_lsd = interpreter_m_lsd.get_input_details()
output_details_m_lsd = interpreter_m_lsd.get_output_details()

model_guitar_detect = YOLO("guitar_detect.pt")
threshold_guitar_detect = 0.3

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('test.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
success, img = cap.read()
h, w, _ = img.shape
detector = HandDetector(detectionCon=0.8, maxHands=2)

mirror_effect = True

canny_lower = 100
canny_upper = 100
angle_global = 45
threshold_prob = 80
lineLength = 500
tolerance = 30

def onChange_lower(val):
    global canny_lower
    canny_lower = val

def onChange_upper(val):
    global canny_upper
    canny_upper = val

def onChange_angle(val):
    global angle_global
    angle_global = val

def onChange_threshold(val):
    global threshold_prob
    threshold_prob = val

def onChange_minLineLength(val):
    global minLineLength
    minLineLength = val

def onChange_maxLineGap(val):
    global maxLineGap
    maxLineGap = val

def onChange_lineLength(val):
    global lineLength
    lineLength = val

def onChange_tolerance(val):
    global tolerance
    tolerance = val

cv2.namedWindow("Edges")
cv2.createTrackbar("lower", "Edges", canny_lower, 1000, onChange_lower)
cv2.createTrackbar("upper", "Edges", canny_upper, 1000, onChange_upper)
cv2.createTrackbar("angle", "Edges", angle_global, 360, onChange_angle)
cv2.createTrackbar("threshold", "Edges", threshold_prob, 500, onChange_threshold)
cv2.createTrackbar("lineLength", "Edges", lineLength, 1000, onChange_lineLength)
cv2.createTrackbar("tolerance", "Edges", tolerance, 1000, onChange_tolerance)

while True:
    success, img = cap.read()

    if mirror_effect:
        img = cv2.flip(img, 1)
    
    img_cut, angle_guitar = guitar_proc(img=img)
    if img_cut is not None and angle_guitar is not None:
        #img_cut = canny_hough_lines(img_cut, canny_lower, canny_upper, threshold_prob, angle_global, lineLength)
        img_cut = canny_hough_line_prob(img_cut, canny_lower, canny_upper, threshold_prob, angle_global, tolerance)
        #img_cut = canny(img_cut)
        #lines = pred_lines(img_cut, interpreter_m_lsd, input_details_m_lsd, output_details_m_lsd, input_shape=[320, 320], score_thr=0.1, dist_thr=20)
        #for line in lines:
        #    x_start, y_start, x_end, y_end = [int(val) for val in line]
        #    cv2.line(img_cut, (x_start, y_start), (x_end, y_end), [0, 0, 255], 2)
       
    else:
        continue
    hand_proc(img=img)

    cv2.imshow("img", img)
    if img_cut is not None:       
        cv2.imshow("img_cut", img_cut)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows() 