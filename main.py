### The following hand-detection code is by cvzone(https://www.computervision.zone/) the video it is taken from: https://www.youtube.com/watch?v=RQ-2JWzNc6k
import sys
import cv2
import math
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import PySimpleGUI as sg

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

    def findHands(self, img, draw_img, draw=True, flipType=True):
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
                    if handType.classification[0].label == "Left":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(draw_img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(draw_img, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)
                    cv2.putText(draw_img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
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

def hand_proc(img_detect, img_draw):
    hands, img = detector.findHands(img_detect, img_draw, draw=True)
    return hands

def object_detect(img, model):
    results = model(img)[0]
    cropped_img = None

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold_detect:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(img, results.names[int(class_id)], (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            
            cropped_img = img[(int(y1) + img_cut_value) : (int(y2) - img_cut_value), (int(x1) + img_cut_value) : (int(x2) - img_cut_value)]
            cut_x = int(x1) + img_cut_value
            cut_y = int(y1) + img_cut_value
            sub_window_coord.append((cut_x, cut_y))
            break 

    return cropped_img

def object_segment(img, model):
    results = model(img)
    mask_rgba = np.zeros_like(img)
    box = None
    center = [0, 0]
    angle = 0
    max_side_length = 0

    if results[0].masks is not None:
        for j, mask in enumerate(results[0].masks.data):
            mask = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_rgba[:,:,0] += mask
            mask_rgba[:,:,1] += mask
            mask_rgba[:,:,2] += mask

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            if max_contour is not None:
                rect = cv2.minAreaRect(max_contour)
                center = [int(rect[0][0]), int(rect[0][1])]
                angle = rect[2]
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                max_side_length = int(max([np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)])) 

                cv2.drawContours(mask_rgba, [box], 0, (0, 0, 255), 2)
                cv2.circle(mask_rgba, center, radius=5, color=(0, 255, 0), thickness=-2)

    return mask_rgba, center, angle, max_side_length

def draw_pre_recorded(img, landmark_list):
    line_list = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
    for line in line_list:
        if len(landmark_list) >= 21:
            cv2.line(img, (landmark_list[line[0]][0], landmark_list[line[0]][1]), (landmark_list[line[1]][0], landmark_list[line[1]][1]), (255, 255, 255), thickness=2)
        if len(landmark_list) == 42:
            cv2.line(img, (landmark_list[line[0] + 21][0], landmark_list[line[0] + 21][1]), (landmark_list[line[1] + 21][0], landmark_list[line[1] + 21][1]), (255, 255, 255), thickness=2)
    for landmark in landmark_list:
        cv2.circle(img, (landmark[0], landmark[1]), radius=5, color=(186, 85, 211), thickness=-2)
    return img

def gui_init():
    global video, pre_recorded_file, mirror_effect, tutor, record, slowing_factor, exit
    layout_gui = [
        [sg.Text("Mirror Effect"), sg.Checkbox("", default=mirror_effect, key="mirror_effect", enable_events=True)],
        [sg.Text("Record"), sg.Checkbox("", default=record, key="record", enable_events=True)],
        [sg.Text("Tutor"), sg.Checkbox("", default=tutor, key="tutor", enable_events=True)],
        [sg.Text("Slowing Factor", key="slowing_factor_text"), sg.Slider(range=(0, 10), default_value=slowing_factor, orientation="h", key="slowing_factor", enable_events=True)],
        [sg.Text("Pre-recorded File", key="pre_recorded_file", visible=tutor), sg.InputText(default_text=pre_recorded_file, key="pre_recorded_file_input_box", visible=tutor), sg.FileBrowse(visible=tutor, key="pre_recorded_file_browse", enable_events=True)],
        [sg.Text("Video", key="video_file", visible=record), sg.InputText(default_text=video, key="video_file_input_box", visible=record), sg.FileBrowse(visible=record, key="video_file_browse", enable_events=True)],
        [sg.Button("Save", key="save", enable_events=True), sg.Button("Exit ARpeggio", key="exit", enable_events=True)]
    ]
    sg.theme("LightGrey5")
    window_gui = sg.Window("Settings", layout_gui)

    while True:
        event, values = window_gui.read()
        if event == sg.WINDOW_CLOSED:
            break
        if event is not None:
            if event == "exit":
                exit = True
                break

            mirror_effect = values["mirror_effect"]
            tutor = values["tutor"]
            record = values["record"]
            slowing_factor = values["slowing_factor"]
            
            window_gui["slowing_factor_text"].update(visible=tutor and not record)
            window_gui["slowing_factor"].update(visible=tutor and not record)
            window_gui["pre_recorded_file"].update(visible=tutor and not record)
            window_gui["pre_recorded_file"].update(visible=tutor and not record)
            window_gui["video_file"].update(visible=record and not tutor)
            window_gui["pre_recorded_file_input_box"].update(visible=tutor and not record)
            window_gui["video_file_input_box"].update(visible=record and not tutor)
            window_gui["pre_recorded_file_browse"].update(visible=tutor and not record)
            window_gui["video_file_browse"].update(visible=record and not tutor)

            if event == "save":
                if tutor and not record:
                    pre_recorded_file = values["pre_recorded_file_input_box"]
                elif record and not tutor:
                    video = values["video_file_input_box"]

    window_gui.close()

def handle_tweaks():
    gui_init()
    if tutor:
        global pre_recorded_file, pre_recorded
        with open(pre_recorded_file, 'r') as text_file:
            data = text_file.read().rstrip(',')
            pre_recorded = [float(x) for x in data.split(',')]

model_guitar_detect = YOLO("guitar_detect.pt")

model_fretboard_detect = YOLO("fretboard_detect.pt")

threshold_detect = 0.3

model_fretboard_seg = YOLO("fretboard_seg.pt")

detector = HandDetector(detectionCon=0.8, maxHands=2)

img_cut_value = 0

video = ""

pre_recorded_file = ""
pre_recorded = []

mirror_effect = False
record = False
tutor = True

counter = 0
step = 128
slowing_factor = 0

exit = False

handle_tweaks()

if record:
    try:
        cap = cv2.VideoCapture(video)
    except FileNotFoundError:
        print("There is no such video file, exiting")
        sys.exit()
else:
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
success, img = cap.read()
h, w, _ = img.shape

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()

    if success is False or exit:
        break

    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
     
    cv2.putText(img, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    img_hands = img.copy()
    
    sub_window_coord = []
    img_guitar = object_detect(img, model_guitar_detect)
    if img_guitar is not None:
        img_fretboard = object_detect(img_guitar, model_fretboard_detect)
        if img_fretboard is not None:
            img_fretboard, center, angle, max_side_length = object_segment(img_fretboard, model_fretboard_seg)
            center[0] += sub_window_coord[0][0] + sub_window_coord[1][0]
            center[1] += sub_window_coord[0][1] + sub_window_coord[1][1]
            if tutor and max_side_length > 0:
                sub_list = pre_recorded[counter : min(counter + step, len(pre_recorded))]
                landmark_list = []
                landmarks = list(sub_list[:-2])
                angle_read = sub_list[-2]
                try:
                    max_side_length_read = sub_list[-1]
                except IndexError:
                    cap.release()
                    cv2.destroyAllWindows() 
                    sys.exit()

                for i in range(0, len(landmarks), 3):
                    x, y, z = sub_list[i:i+3]

                    x += center[0]
                    y += center[1]

                    x_rotated = (x - center[0]) * np.cos(np.deg2rad(angle - angle_read)) - (y - center[1]) * np.sin(np.deg2rad(angle - angle_read)) + center[0]
                    y_rotated = (x - center[0]) * np.sin(np.deg2rad(angle - angle_read)) + (y - center[1]) * np.cos(np.deg2rad(angle - angle_read)) + center[1]

                    x *= max_side_length/max_side_length_read
                    y *= max_side_length/max_side_length_read

                    landmark_list.append([int(x_rotated), int(y_rotated), int(z)])
                img = draw_pre_recorded(img, landmark_list)
                if slowing_factor != 0:
                    if int(time.time()) % slowing_factor == 0:
                        counter += step
                else:
                    counter += step

    else:
        continue

    hands = hand_proc(img_detect=img_hands, img_draw=img)
    if hands:
        if record and len(hands) == 2 and max_side_length > 0:
            write_data = []
            for hand in hands:
                for landmark in hand["lmList"]:
                    #r = np.sqrt(pow(landmark[0] - center[0], 2) + pow(landmark[1] - center[1], 2))
                    #theta = np.arctan2(landmark[1] - center[1], landmark[0] - center[0])
                    #write_data.extend([r, theta, landmark[2]])
                    write_data.extend([landmark[0] - center[0], landmark[1] - center[1], landmark[2]])
            write_data.extend([angle, max_side_length])
            with open(video.replace(".mp4", ".txt"), 'a') as text_file:
                text_file.write(','.join(str(x) for x in write_data))
                text_file.write(',')


    if mirror_effect:
        img = cv2.flip(img, 1)
        img_guitar = cv2.flip(img_guitar, 1)
        img_fretboard = cv2.flip(img_fretboard, 1)

    cv2.imshow("img", img)
    if img_guitar is not None:       
        cv2.imshow("img_guitar", img_guitar)
    if img_fretboard is not None:
        cv2.imshow("img_fretboard", img_fretboard)
    
    if cv2.waitKey(1) == 27:
        handle_tweaks()


cap.release()
cv2.destroyAllWindows() 