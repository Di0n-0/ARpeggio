### The following hand-detection code is by cvzone(https://www.computervision.zone/) the video it is taken from: https://www.youtube.com/watch?v=RQ-2JWzNc6k
import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO
import decipher as deci
import gui

def gui_init():
    global file_path_tab, dev_mode, show_fretboard, mirror_effect, ascii_tutor, slowing_factor, alpha, exit
    file_path_tab, dev_mode, show_fretboard, mirror_effect, ascii_tutor, slowing_factor, alpha, exit = gui.main()
    alpha /= 10
    if exit:
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()
        sys.exit()

def handle_tweaks():
    global deciphered_point_data
    gui_init()
    if ascii_tutor and not len(deciphered_point_data):
        try:
            deciphered_point_data = deci.main(file_path_tab)
        except FileNotFoundError:
            print("There is no such text file, exiting...")
            sys.exit()

def object_segment(img, model):
    results = model(img, device="cpu")
    mask_rgba = np.zeros_like(img)
    intersections = []
    guitar_strings = []

    if results[0].masks is not None:
        mask = results[0].masks.data[0]
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

            angle = rect[2]
            if not angle:
                return intersections, guitar_strings
            
            height_rect = rect[1][1]
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            sorted_points = box[np.argsort(box[:, 1])[::-1]]
            point1, point2 = sorted_points[:2]
            point1 = list(sorted_points[0])
            point1[0] = mask_rgba.shape[0] - point1[0]
            point2 = list(sorted_points[1])
            point2[0] = mask_rgba.shape[0] - point2[0]

            tan = np.tan(np.deg2rad(angle))
            for i in range (1,7):
                guitar_strings.append((tan, (i*(point2[0]-tan*point2[1]) + (7-i)*(point1[0]-tan*point1[1]))/7))
            
            fret_count = 19
            scale_length = height_rect*np.power(2, fret_count/12)/(np.power(2, fret_count/12)-1)
            fret_cal = lambda x : height_rect-(scale_length - scale_length/np.power(2, x/12))
            fret_distance = [fret_cal(i) for i in range(0, fret_count + 1)]
            guitar_frets = []
            cot = 1/tan
            for d in fret_distance:
                guitar_frets.append((-cot, (point1[0]+cot*point1[1]) - d*(np.sqrt(1 + np.square(cot)))))
            
            guitar_fret_mids = []
            for i in range(0, len(guitar_frets) - 1):
                guitar_fret_mids.append((-cot, (guitar_frets[i][1]+guitar_frets[i+1][1])/2))
            
            
            for guitar_string in guitar_strings:
                for guitar_fret_mid in guitar_fret_mids:
                    intersections.append(((guitar_fret_mid[1]-guitar_string[1])/(guitar_string[0]-guitar_fret_mid[0]), guitar_string[0]*(guitar_fret_mid[1]-guitar_string[1])/(guitar_string[0]-guitar_fret_mid[0]) + guitar_string[1]))
            for img_draw in [mask_rgba, img]:
                cv2.drawContours(img_draw, [box], 0, (0, 0, 255), 2)
            if show_fretboard:
                for guitar_string in guitar_strings:
                    try:
                        cv2.line(img_draw, (int(mask_rgba.shape[0] - (0 * guitar_string[0] + guitar_string[1])), 0), (int(mask_rgba.shape[0] - (mask_rgba.shape[1] * guitar_string[0] + guitar_string[1])), mask_rgba.shape[1]), color=(255, 255, 255), thickness=2)
                    except cv2.error:
                        continue
                """
                for guitar_fret in guitar_frets:
                    cv2.line(img_draw, (int(mask_rgba.shape[0] - (0 * guitar_fret[0] + guitar_fret[1])), 0), (int(mask_rgba.shape[0] - (mask_rgba.shape[1] * guitar_fret[0] + guitar_fret[1])), mask_rgba.shape[1]), color=(0, 255, 255), thickness=2)
                for guitar_fret_mid in guitar_fret_mids:
                    cv2.line(img_draw, (int(mask_rgba.shape[0] - (0 * guitar_fret_mid[0] + guitar_fret_mid[1])), 0), (int(mask_rgba.shape[0] - (mask_rgba.shape[1] * guitar_fret_mid[0] + guitar_fret_mid[1])), mask_rgba.shape[1]), color=(255, 0, 255), thickness=2)
                """
                for intersection in intersections:
                    cv2.circle(img_draw, (int(mask_rgba.shape[0] - intersection[1]), int(intersection[0])), radius=3, color=(250, 150, 0), thickness=-1)
                    
    return intersections, guitar_strings
  
def ascii_tutor_points(img, intersections, guitar_strings):
    global counter_tab, previous_time
    overlay = img.copy()

    for point_list in deciphered_point_data[counter_tab]:
        point_index = point_list[0]
        point_color = point_list[2]
        if point_index < 0:
            try:
                cv2.line(overlay, (int(img.shape[0] - (0 * guitar_strings[-point_index - 1][0] + guitar_strings[-point_index - 1][1])), 0), (int(img.shape[0] - (img.shape[1] * guitar_strings[-point_index - 1][0] + guitar_strings[-point_index - 1][1])), img.shape[1]), color=point_color, thickness=5)
            except cv2.error:
                continue
        else:
            point_index -= 1
            cv2.circle(overlay, (int(img.shape[0] - intersections[point_index][1]), int(intersections[point_index][0])), radius=7, color=point_color, thickness=-1)
    
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    current_time = int(time.time())
    if slowing_factor != 0:
        if previous_time != current_time and current_time % slowing_factor == 0:
            previous_time = current_time
            if counter_tab != len(deciphered_point_data) - 1:
                counter_tab += 1
    else:
        if counter_tab != len(deciphered_point_data) - 1:
            counter_tab += 1
    return img

def proc_guitar(img):
    intersections, guitar_strings = object_segment(img, model_fretboard_seg)
    if ascii_tutor and len(intersections) != 0 and len(guitar_strings) != 0:
        img = ascii_tutor_points(img, intersections, guitar_strings)
    return img

def main():
    global cap
    handle_tweaks()

   
    cap = cv2.VideoCapture(0)#0
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previousTime = 0
    currentTime = 0

    while ascii_tutor:
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

        img = proc_guitar(img)
        if img is None:
            continue

        if mirror_effect:
            img = cv2.flip(img, 1)

        try:
            cv2.imshow("ARpeggio", img)
        except cv2.error:
            print(img)
    cap.release()
    cv2.destroyAllWindows() 

model_fretboard_seg = YOLO("../models/fretboard_seg.pt")

img_cut_value = 0

file_path_tab = "ASCII Tablature"
deciphered_point_data = []

previous_time = int(time.time())

dev_mode = True
mirror_effect = False
ascii_tutor = False
show_fretboard = True
exit = False

alpha = 0.7

counter_tab = 0
slowing_factor = 0

cap = None

if __name__ == "__main__":
    main()
