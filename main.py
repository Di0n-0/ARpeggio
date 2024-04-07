### The following hand-detection code is by cvzone(https://www.computervision.zone/) the video it is taken from: https://www.youtube.com/watch?v=RQ-2JWzNc6k
import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import PySimpleGUI as sg
import decipher as deci

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
                    try:
                        self.mpDraw.draw_landmarks(draw_img, handLms, self.mpHands.HAND_CONNECTIONS)
                    except AttributeError:
                        print(draw_img)
                        continue
                    cv2.rectangle(draw_img, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)
                    cv2.putText(draw_img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return allHands

def gui_init():
    global file_path, file_path_tab, dev_mode, show_hands, show_fretboard, mirror_effect, ascii_tutor, tutor, record, slowing_factor, draw_index_right, draw_index_left, alpha
    sg.theme("LightGrey5")
    file_types = [("Supported Files", "*.mp4 *.txt")]
    layout_gui = [
        [sg.Text("Developer Mode"), sg.Checkbox("", default=dev_mode, key="dev_mode", enable_events=True)],
        [sg.Text("Hands"), sg.Checkbox("", default=show_hands, key="hands", enable_events=True)],
        [sg.Text("Fretboard"), sg.Checkbox("", default=show_fretboard, key="fretboard", enable_events=True)],
        [sg.Text("Mirror Effect"), sg.Checkbox("", default=mirror_effect, key="mirror_effect", enable_events=True)],
        [sg.Text("Slowing Factor", key="slowing_factor_text"), sg.Slider(range=(0, 10), default_value=slowing_factor, orientation="h", key="slowing_factor", enable_events=True)],
        [sg.Text("Alpha Value", key="alpha_value_text"), sg.Slider(range=(0, 10), default_value=alpha*10, orientation="h", key="alpha", enable_events=True)],
        [sg.InputText(default_text=file_path, key="-FILE-", enable_events=True), sg.FileBrowse(file_types=file_types)],
        [sg.InputText(default_text=file_path_tab, key="-FILE_TAB-", enable_events=True), sg.FileBrowse(file_types=(("Text Files", "*.txt"),))],
        [sg.Text("Hand 0"), sg.InputText(default_text=draw_index_right, key="right_hand", enable_events=True)],
        [sg.Text("Hand 1"), sg.InputText(default_text=draw_index_left, key="left_hand", enable_events=True)],
        [sg.Image("hand_landmarks.png")],
        [sg.Button("Exit ARpeggio", key="exit", enable_events=True)]
    ]
    window_gui = sg.Window("Settings", layout_gui)

    while True:
        event, values = window_gui.read()
        if event == sg.WINDOW_CLOSED:
            break
        if event is not None:
            if event == "exit":
                if cap is not None:
                    cap.release()
                    cv2.destroyAllWindows()
                window_gui.close()
                sys.exit()
            elif event == "-FILE-":
                file_path = values["-FILE-"]  
            elif event == "-FILE_TAB-":
                file_path_tab = values["-FILE_TAB-"]
            elif event == "right_hand":
                draw_index_right = values["right_hand"]
            elif event == "left_hand":
                draw_index_left = values["left_hand"]
            dev_mode = values["dev_mode"]
            show_hands = values["hands"]
            show_fretboard = values["fretboard"]
            mirror_effect = values["mirror_effect"]
            slowing_factor = values["slowing_factor"]
            alpha = values["alpha"] / 10
            if file_path.endswith(".txt"):
                tutor = True
                record = False
                ascii_tutor = False
            elif file_path.endswith(".mp4"):
                tutor = False
                record = True
                ascii_tutor = False
            else:
                tutor = False
                record = False
                if file_path_tab.endswith(".txt"):
                    ascii_tutor = True

    window_gui.close()

def handle_tweaks():
    global pre_recorded, deciphered_point_data
    gui_init()
    if tutor:
        try:
            with open(file_path, 'r') as text_file:
                data = text_file.read().rstrip(',')
                pre_recorded = [float(x) for x in data.split(',')]
        except FileNotFoundError:
            print("There is no such pre-recorded file, exiting")
            sys.exit()
    elif ascii_tutor:
        try:
            deciphered_point_data = deci.main(file_path_tab)
        except FileNotFoundError:
            print("There is no such pre-recorded file, exiting")
            sys.exit()

def object_detect(img, model, sub_window_coord):
    results = model(img)[0]
    cropped_img = None

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold_detect:
            if dev_mode:
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
    center = [0, 0]
    angle = 0
    width_rect = 0
    height_rect = 0

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
                width_rect = rect[1][0]
                height_rect = rect[1][1]
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                sorted_points = box[np.argsort(box[:, 1])[::-1]]
                point1, point2 = sorted_points[:2]
                point1 = list(sorted_points[0])
                point1[0] = mask_rgba.shape[0] - point1[0]
                point2 = list(sorted_points[1])
                point2[0] = mask_rgba.shape[0] - point2[0]
                guitar_strings = []
                tan = np.tan(np.deg2rad(angle))
                for i in range (1,7):
                    guitar_strings.append((tan, (i*(point2[0]-tan*point2[1]) + (7-i)*(point1[0]-tan*point1[1]))/7))
                
                fret_count = 19
                fret_cal = lambda x : height_rect-(height_rect*np.power(2, fret_count/12)/(np.power(2, fret_count/12)-1) - height_rect*np.power(2, fret_count/12)/(np.power(2, fret_count/12)-1)/np.power(2, x/12))
                fret_distance = [fret_cal(i) for i in range(0, fret_count + 1)]
                guitar_frets = []
                cot = 1/tan
                for d in fret_distance:
                    guitar_frets.append((-cot, (point1[0]+cot*point1[1]) - d*(np.sqrt(1 + np.square(cot)))))
                
                guitar_fret_mids = []
                for i in range(0, len(guitar_frets) - 1):
                    guitar_fret_mids.append((-cot, (guitar_frets[i][1]+guitar_frets[i+1][1])/2))
                
                intersections = []
                for guitar_string in guitar_strings:
                    for guitar_fret_mid in guitar_fret_mids:
                        intersections.append(((guitar_fret_mid[1]-guitar_string[1])/(guitar_string[0]-guitar_fret_mid[0]), guitar_string[0]*(guitar_fret_mid[1]-guitar_string[1])/(guitar_string[0]-guitar_fret_mid[0]) + guitar_string[1]))

                if dev_mode:
                    for img_draw in [mask_rgba, img]:
                        cv2.drawContours(img_draw, [box], 0, (0, 0, 255), 2)
                        cv2.circle(img_draw, center, radius=5, color=(0, 0, 255), thickness=-2)
                    if show_fretboard:
                        """
                        for guitar_string in guitar_strings:
                            cv2.line(img_draw, (int(mask_rgba.shape[0] - (0 * guitar_string[0] + guitar_string[1])), 0), (int(mask_rgba.shape[0] - (mask_rgba.shape[1] * guitar_string[0] + guitar_string[1])), mask_rgba.shape[1]), color=(255, 255, 255), thickness=2)
                        for guitar_fret in guitar_frets:
                            cv2.line(img_draw, (int(mask_rgba.shape[0] - (0 * guitar_fret[0] + guitar_fret[1])), 0), (int(mask_rgba.shape[0] - (mask_rgba.shape[1] * guitar_fret[0] + guitar_fret[1])), mask_rgba.shape[1]), color=(0, 255, 255), thickness=2)
                        for guitar_fret_mid in guitar_fret_mids:
                            cv2.line(img_draw, (int(mask_rgba.shape[0] - (0 * guitar_fret_mid[0] + guitar_fret_mid[1])), 0), (int(mask_rgba.shape[0] - (mask_rgba.shape[1] * guitar_fret_mid[0] + guitar_fret_mid[1])), mask_rgba.shape[1]), color=(255, 0, 255), thickness=2)
                        """
                        for intersection in intersections:
                            cv2.circle(img_draw, (int(mask_rgba.shape[0] - intersection[1]), int(intersection[0])), radius=3, color=(250, 150, 0), thickness=-1)

    return mask_rgba, center, angle, width_rect, height_rect, intersections

def draw_pre_recorded(img, landmark_list):
    try:
        index_list_right = [int(num.strip()) for num in draw_index_right.split(',')]
    except ValueError:
        index_list_right = None
    try:
        index_list_left = [int(num.strip()) for num in draw_index_left.split(',')]
    except ValueError:
        index_list_left = None
    
    if show_hands:
        line_list = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
        line_list_right = []
        line_list_left = []
        for line in line_list:
            if index_list_right is not None:
                if line[0] in index_list_right and line[1] in index_list_right:
                    line_list_right.append(line)
            if index_list_left is not None:
                if line[0] in index_list_left and line[1] in index_list_left:
                    line_list_left.append(line)
        overlay = img.copy()
        if index_list_left is not None:
            for line in line_list_left:
                cv2.line(overlay, (landmark_list[line[0]][0], landmark_list[line[0]][1]), (landmark_list[line[1]][0], landmark_list[line[1]][1]), (255, 255, 255, 128), thickness=2)
        if index_list_right is not None:
            for line in line_list_right:
                cv2.line(overlay, (landmark_list[line[0] + 21][0], landmark_list[line[0] + 21][1]), (landmark_list[line[1] + 21][0], landmark_list[line[1] + 21][1]), (255, 255, 255, 128), thickness=2)
        if index_list_left is not None:
            for index in index_list_left:
                cv2.circle(overlay, (landmark_list[index][0], landmark_list[index][1]), radius=4, color=(186, 85, 211, 128), thickness=-1)
        if index_list_right is not None:
            for index in index_list_right:
                cv2.circle(overlay, (landmark_list[index + 21][0], landmark_list[index + 21][1]), radius=4, color=(186, 85, 211, 128), thickness=-1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def tutor_hands(img, center, angle, width_rect, height_rect):
  global counter
  if width_rect > 0 and height_rect > 0:
    sub_list = pre_recorded[counter : min(counter + step, len(pre_recorded))]
    landmark_list = []
    landmarks = list(sub_list[:-3])
    angle_read = sub_list[-3]
    width_rect_read = sub_list[-2]
    try:
        height_rect_read = sub_list[-1]
    except IndexError:
        cap.release()
        cv2.destroyAllWindows() 
        sys.exit()

    for i in range(0, len(landmarks), 3):
        x, y, z = sub_list[i:i+3]

        x *= width_rect / width_rect_read
        y *= height_rect / height_rect_read

        x += center[0]
        y += center[1]

        x_rotated = (x - center[0]) * np.cos(np.deg2rad(angle - angle_read)) - (y - center[1]) * np.sin(np.deg2rad(angle - angle_read)) + center[0]
        y_rotated = (x - center[0]) * np.sin(np.deg2rad(angle - angle_read)) + (y - center[1]) * np.cos(np.deg2rad(angle - angle_read)) + center[1]

        landmark_list.append([round(x_rotated), round(y_rotated), round(z)])
    img = draw_pre_recorded(img, landmark_list)
    if slowing_factor != 0:
        if int(time.time()) % slowing_factor == 0:
            counter += step
    else:
        counter += step
    return img
  
def ascii_tutor_points(img, img_fretboard, sub_window_coord, intersections):
    overlay = img.copy()
    """
    for point_index in deciphered_point_data:
        if not point_index:
            continue
        point_index = point_index - 1
        cv2.circle(overlay, (int(overlay.shape[0] - intersections[point_index][1]), int(intersections[point_index][0])), radius=3, color=(0, 150, 250), thickness=-1)
    """
    cv2.circle(overlay, (int(img_fretboard.shape[0] - intersections[4][1] + (sub_window_coord[0][0] + sub_window_coord[1][0])), int(intersections[4][0] + (sub_window_coord[0][1] + sub_window_coord[1][1]))), radius=3, color=(0, 150, 250), thickness=-1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def proc_guitar(img):
    sub_window_coord = []
    img_guitar = object_detect(img, model_guitar_detect, sub_window_coord)
    if img_guitar is not None:
        img_fretboard = object_detect(img_guitar, model_fretboard_detect, sub_window_coord)
        if img_fretboard is not None:
            img_fretboard, center, angle, width_rect, height_rect, intersections = object_segment(img_fretboard, model_fretboard_seg)
            center[0] += sub_window_coord[0][0] + sub_window_coord[1][0]
            center[1] += sub_window_coord[0][1] + sub_window_coord[1][1]
            if tutor:
                img = tutor_hands(img, center, angle, width_rect, height_rect)
            elif ascii_tutor:
                img = ascii_tutor_points(img, img_fretboard, sub_window_coord,intersections)
            return img, img_guitar, img_fretboard, center, angle, width_rect, height_rect
    return None

def proc_hands(img_detect, img_draw):
    hands = detector.findHands(img_detect, img_draw, draw=dev_mode)
    return hands

def record_hands(hands, center, angle, width_rect, height_rect):
    if hands:
        if len(hands) == 2 and width_rect > 0 and height_rect > 0:
            write_data = []
            for hand in hands:
                for landmark in hand["lmList"]:
                    write_data.extend([landmark[0] - center[0], landmark[1] - center[1], landmark[2]])
            write_data.extend([angle, width_rect, height_rect])
            with open(file_path.replace(".mp4", ".txt"), 'a') as text_file:
                text_file.write(','.join(str(x) for x in write_data))
                text_file.write(',')

def main():
    global cap
    handle_tweaks()

    if record:
        try:
            cap = cv2.VideoCapture(file_path)
        except FileNotFoundError:
            print("There is no such video file, exiting")
            sys.exit()
    else:
        cap = cv2.VideoCapture("test2.mp4")#0 #test1.mp4
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previousTime = 0
    currentTime = 0

    while tutor or record or ascii_tutor:
        success, img = cap.read()
        if success is False:
            break
 
        if cv2.waitKey(1) == 27:
            handle_tweaks()

        currentTime = time.time()
        fps = 1 / (currentTime-previousTime)
        previousTime = currentTime
        
        if dev_mode:
            cv2.putText(img, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        img_hands = img.copy()

        output = proc_guitar(img)
        if output is None:
            continue
        img, img_guitar, img_fretboard, center, angle, width_rect, height_rect = output

        hands = proc_hands(img_detect=img_hands, img_draw=img)
        if record:
            record_hands(hands, center, angle, width_rect, height_rect)

        if mirror_effect:
            img = cv2.flip(img, 1)
            img_guitar = cv2.flip(img_guitar, 1)
            img_fretboard = cv2.flip(img_fretboard, 1)

        try:
            cv2.imshow("ARpeggio", img)
        except cv2.error:
            print(img)
        if img_guitar is not None and dev_mode:       
            cv2.imshow("img_guitar", img_guitar)
        if img_fretboard is not None and dev_mode:
            cv2.imshow("img_fretboard", img_fretboard)

    cap.release()
    cv2.destroyAllWindows() 

model_guitar_detect = YOLO("guitar_detect.pt")
model_fretboard_detect = YOLO("fretboard_detect.pt")
threshold_detect = 0.8

model_fretboard_seg = YOLO("fretboard_seg.pt")

detector = HandDetector(detectionCon=0.8, maxHands=2)

img_cut_value = 0

file_path = "A video to be analysed or a text file of an analysed video"
file_path_tab = "ASCII Tablature"
pre_recorded = []
deciphered_point_data = []

dev_mode = True
mirror_effect = False
record = False
tutor = False
ascii_tutor = False
show_hands = True
show_fretboard = True

draw_index_right = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
draw_index_left = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"

alpha = 0.7

counter = 0
step = 129
slowing_factor = 0

cap = None

if __name__ == "__main__":
    main()
