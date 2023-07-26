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

def object_detect(img, model):
    results = model(img)[0]
    cropped_img = None
    angle = None

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if score > threshold_guitar_detect:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(img, results.names[int(class_id)], (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            
            cropped_img = img[(int(y1) + img_cut_value) : (int(y2) - img_cut_value), (int(x1) + img_cut_value) : (int(x2) - img_cut_value)]
            break 

    return cropped_img, angle

def object_segment(img, model):
    results = model(img)
    mask_rgba = np.zeros_like(img)
    
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
                hull = cv2.convexHull(max_contour)
                peri = cv2.arcLength(max_contour, True)
                approx_curve = cv2.approxPolyDP(max_contour, epsilon=0.002 * peri, closed=True)
                rect = cv2.minAreaRect(approx_curve)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                for vertex in hull:
                    x, y = vertex[0]
                    cv2.circle(mask_rgba, (x, y), radius=5, color=(0, 255, 0), thickness=-2)
                cv2.drawContours(mask_rgba, [hull], -1, (255,0,255), thickness=4)
                cv2.drawContours(mask_rgba, [box], 0, (0, 0, 255), 2)
                left_edge = min(box[:, 0])
                right_edge = max(box[:, 0])
                top_edge = min(box[:, 1])
                bottom_edge = max(box[:, 1])

    return mask_rgba

def canny_hough_line_prob(img, canny_lower, canny_upper, threshold=80):
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

            if not (abs(slope) < abs(np.tan(np.deg2rad(angle_global + tolerance))) and abs(np.tan(np.deg2rad(angle_global - tolerance))) < abs(slope)): 
                continue

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

        cv2.line(cdst, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2, cv2.LINE_AA)
    
    return cdst
    
def canny(img):
    gray_scale = img.copy()
    gray_scale = cv2.cvtColor(gray_scale, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_scale, canny_lower, canny_upper, None, 3)
    img_return = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return img_return

def pred_lines(image, interpreter, input_details, output_details, input_shape=[512, 512], score_thr=0.10, dist_thr_min = 20, dist_thr_max=40):
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
        if score > score_thr and (distance > dist_thr_min and distance < dist_thr_max):
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end

            if x_end - x_start != 0:
                slope = (y_end - y_start) / (x_end - x_start)
            else:
                slope = np.inf

            if not (abs(slope) < abs(np.tan(np.deg2rad(angle_global + tolerance))) and abs(np.tan(np.deg2rad(angle_global - tolerance))) < abs(slope)): 
                continue

            segments_list.append([x_start, y_start, x_end, y_end])
    
    lines = 2 * np.array(segments_list) # 256 > 512
    if lines.ndim == 2:
        lines[:,0] = lines[:,0] * w_ratio
        lines[:,1] = lines[:,1] * h_ratio
        lines[:,2] = lines[:,2] * w_ratio
        lines[:,3] = lines[:,3] * h_ratio
        return lines
    
    return None

def pred_squares(image,
                 interpreter,
                 input_details,
                 output_details,
                 input_shape=[512, 512],
                 params={'score': 0.01,
                         'outside_ratio': 0.28,
                         'inside_ratio': 0.45,
                         'w_overlap': 0.0,
                         'w_degree': 1.95,
                         'w_length': 0.0,
                         'w_area': 1.86,
                         'w_center': 0.14}):
    h, w, _ = image.shape
    original_shape = [h, w]

    resized_image = np.concatenate([cv2.resize(image, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA), np.ones([input_shape[0], input_shape[1], 1])], axis=-1)
    batch_image = np.expand_dims(resized_image, axis=0).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], batch_image)
    interpreter.invoke()

    pts = interpreter.get_tensor(output_details[0]['index'])[0]
    pts_score = interpreter.get_tensor(output_details[1]['index'])[0]
    vmap = interpreter.get_tensor(output_details[2]['index'])[0]
    
    start = vmap[:,:,:2] # (x, y)
    end = vmap[:,:,2:] # (x, y)
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    junc_list = []
    segments_list = []
    for junc, score in zip(pts, pts_score):
        y, x = junc
        distance = dist_map[y, x]
        if score > params['score'] and distance > 20.0:
            junc_list.append([x, y])
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            d_arrow = 1.0
            x_start = x + d_arrow * disp_x_start
            y_start = y + d_arrow * disp_y_start
            x_end = x + d_arrow * disp_x_end
            y_end = y + d_arrow * disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])
            
    segments = np.array(segments_list)
    
    ####### post processing for squares
    # 1. get unique lines
    point = np.array([[0, 0]])
    point = point[0]
    start = segments[:,:2]
    end = segments[:,2:]
    diff = start - end
    a = diff[:, 1]
    b = -diff[:, 0]
    c = a * start[:,0] + b * start[:,1]

    d = np.abs(a * point[0] + b * point[1] - c) / np.sqrt(a ** 2 + b ** 2 + 1e-10)
    theta = np.arctan2(diff[:,0], diff[:,1]) * 180 / np.pi
    theta[theta < 0.0] += 180
    hough = np.concatenate([d[:,None], theta[:,None]], axis=-1)

    d_quant = 1
    theta_quant = 2
    hough[:,0] //= d_quant
    hough[:,1] //= theta_quant
    _, indices, counts = np.unique(hough, axis=0, return_index=True, return_counts=True)
    
    acc_map = np.zeros([512 // d_quant + 1, 360 // theta_quant + 1], dtype='float32')
    idx_map = np.zeros([512 // d_quant + 1, 360 // theta_quant + 1], dtype='int32') - 1
    yx_indices = hough[indices,:].astype('int32')
    acc_map[yx_indices[:,0], yx_indices[:,1]] = counts
    idx_map[yx_indices[:,0], yx_indices[:,1]] = indices
    
    acc_map_np = acc_map
    acc_map = acc_map[None,:,:,None]
    
    ### fast suppression using tensorflow op
    acc_map = tf.constant(acc_map, dtype=tf.float32)
    max_acc_map = tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=1, padding='same')(acc_map)
    acc_map = acc_map * tf.cast(tf.math.equal(acc_map, max_acc_map), tf.float32)
    flatten_acc_map = tf.reshape(acc_map, [1, -1])
    topk_values, topk_indices = tf.math.top_k(flatten_acc_map, k=len(pts))
    _, h, w, _ = acc_map.shape
    y = tf.expand_dims(topk_indices // w, axis=-1)
    x = tf.expand_dims(topk_indices % w, axis=-1)
    yx = tf.concat([y, x], axis=-1)
    ###

    yx = yx[0].numpy()
    indices = idx_map[yx[:,0], yx[:,1]]
    topk_values = topk_values.numpy()[0]
    basis = 5 // 2

    merged_segments = []
    for yx_pt, max_indice, value in zip(yx, indices, topk_values):
        y, x = yx_pt
        if max_indice == -1 or value == 0:
            continue
        segment_list = []
        for y_offset in range(-basis, basis+1):
            for x_offset in range(-basis, basis+1):
                indice = idx_map[y+y_offset,x+x_offset]
                cnt = int(acc_map_np[y+y_offset,x+x_offset])
                if indice != -1:
                    segment_list.append(segments[indice])
                if cnt > 1:
                    check_cnt = 1
                    current_hough = hough[indice]
                    for new_indice, new_hough in enumerate(hough):
                        if (current_hough == new_hough).all() and indice != new_indice:
                            segment_list.append(segments[new_indice])
                            check_cnt += 1
                        if check_cnt == cnt:
                            break
        group_segments = np.array(segment_list).reshape([-1, 2])
        sorted_group_segments = np.sort(group_segments, axis=0)
        x_min, y_min = sorted_group_segments[0,:]
        x_max, y_max = sorted_group_segments[-1,:]

        deg = theta[max_indice]
        if deg >= 90:
            merged_segments.append([x_min, y_max, x_max, y_min])
        else:
            merged_segments.append([x_min, y_min, x_max, y_max])

    # 2. get intersections
    new_segments = np.array(merged_segments) # (x1, y1, x2, y2)
    start = new_segments[:,:2] # (x1, y1)
    end = new_segments[:,2:] # (x2, y2)
    new_centers = (start + end) / 2.0
    diff = start - end
    dist_segments = np.sqrt(np.sum(diff ** 2, axis=-1))

    # ax + by = c
    a = diff[:,1]
    b = -diff[:,0]
    c = a * start[:,0] + b * start[:,1]
    pre_det = a[:,None] * b[None,:]
    det = pre_det - np.transpose(pre_det)

    pre_inter_y = a[:,None] * c[None,:]
    inter_y = (pre_inter_y - np.transpose(pre_inter_y)) / (det + 1e-10)
    pre_inter_x = c[:,None] * b[None,:]
    inter_x = (pre_inter_x - np.transpose(pre_inter_x)) / (det + 1e-10)
    inter_pts = np.concatenate([inter_x[:,:,None], inter_y[:,:,None]], axis=-1).astype('int32')
    
    # 3. get corner information
    # 3.1 get distance
    '''
    dist_segments:
        | dist(0), dist(1), dist(2), ...|
    dist_inter_to_segment1:
        | dist(inter,0), dist(inter,0), dist(inter,0), ... |
        | dist(inter,1), dist(inter,1), dist(inter,1), ... |
        ...
    dist_inter_to_semgnet2:
        | dist(inter,0), dist(inter,1), dist(inter,2), ... |
        | dist(inter,0), dist(inter,1), dist(inter,2), ... |
        ...
    '''

    dist_inter_to_segment1_start = np.sqrt(np.sum(((inter_pts - start[:,None,:]) ** 2), axis=-1, keepdims=True)) # [n_batch, n_batch, 1]
    dist_inter_to_segment1_end = np.sqrt(np.sum(((inter_pts - end[:,None,:]) ** 2), axis=-1, keepdims=True)) # [n_batch, n_batch, 1]
    dist_inter_to_segment2_start = np.sqrt(np.sum(((inter_pts - start[None,:,:]) ** 2), axis=-1, keepdims=True)) # [n_batch, n_batch, 1]
    dist_inter_to_segment2_end = np.sqrt(np.sum(((inter_pts - end[None,:,:]) ** 2), axis=-1, keepdims=True)) # [n_batch, n_batch, 1]
    
    # sort ascending
    dist_inter_to_segment1 = np.sort(np.concatenate([dist_inter_to_segment1_start, dist_inter_to_segment1_end], axis=-1), axis=-1) # [n_batch, n_batch, 2]
    dist_inter_to_segment2 = np.sort(np.concatenate([dist_inter_to_segment2_start, dist_inter_to_segment2_end], axis=-1), axis=-1) # [n_batch, n_batch, 2]

    # 3.2 get degree
    inter_to_start = new_centers[:,None,:] - inter_pts
    deg_inter_to_start = np.arctan2(inter_to_start[:,:,1], inter_to_start[:,:,0]) * 180 / np.pi
    deg_inter_to_start[deg_inter_to_start < 0.0] += 360
    inter_to_end = new_centers[None,:,:] - inter_pts
    deg_inter_to_end = np.arctan2(inter_to_end[:,:,1], inter_to_end[:,:,0]) * 180 / np.pi
    deg_inter_to_end[deg_inter_to_end < 0.0] += 360
    
    '''
    0 -- 1
    |    |
    3 -- 2
    '''
    # rename variables
    deg1_map, deg2_map = deg_inter_to_start, deg_inter_to_end
    # sort deg ascending
    deg_sort = np.sort(np.concatenate([deg1_map[:,:,None], deg2_map[:,:,None]], axis=-1), axis=-1)
    
    deg_diff_map = np.abs(deg1_map - deg2_map)
    # we only consider the smallest degree of intersect
    deg_diff_map[deg_diff_map > 180] = 360 - deg_diff_map[deg_diff_map > 180]
    
    # define available degree range
    deg_range = [60, 120]
    
    corner_dict = {corner_info: [] for corner_info in range(4)}
    inter_points = []
    for i in range(inter_pts.shape[0]):
        for j in range(i + 1, inter_pts.shape[1]):
            # i, j > line index, always i < j
            x, y = inter_pts[i, j, :]
            deg1, deg2 = deg_sort[i, j, :]
            deg_diff = deg_diff_map[i, j]
            
            check_degree = deg_diff > deg_range[0] and deg_diff < deg_range[1]

            outside_ratio = params['outside_ratio'] # over ratio >>> drop it!
            inside_ratio = params['inside_ratio'] # over ratio >>> drop it!
            check_distance = ((dist_inter_to_segment1[i,j,1] >= dist_segments[i] and \
                                 dist_inter_to_segment1[i,j,0] <= dist_segments[i] * outside_ratio) or \
                                (dist_inter_to_segment1[i,j,1] <= dist_segments[i] and \
                                 dist_inter_to_segment1[i,j,0] <= dist_segments[i] * inside_ratio)) and \
                             ((dist_inter_to_segment2[i,j,1] >= dist_segments[j] and \
                                 dist_inter_to_segment2[i,j,0] <= dist_segments[j] * outside_ratio) or \
                                (dist_inter_to_segment2[i,j,1] <= dist_segments[j] and \
                                 dist_inter_to_segment2[i,j,0] <= dist_segments[j] * inside_ratio))

            if check_degree and check_distance:
                corner_info = None

                if (deg1 >= 0 and deg1 <= 45 and deg2 >=45 and deg2 <= 120) or \
                     (deg2 >= 315 and deg1 >= 45 and deg1 <= 120):
                    corner_info, color_info = 0, 'blue'
                elif (deg1 >= 45 and deg1 <= 125 and deg2 >= 125 and deg2 <= 225):
                    corner_info, color_info = 1, 'green'
                elif (deg1 >= 125 and deg1 <= 225 and deg2 >= 225 and deg2 <= 315):
                    corner_info, color_info = 2, 'black'
                elif (deg1 >= 0 and deg1 <= 45 and deg2 >= 225 and deg2 <= 315) or \
                     (deg2 >= 315 and deg1 >= 225 and deg1 <= 315):
                    corner_info, color_info = 3, 'cyan'
                else:
                    corner_info, color_info = 4, 'red' # we don't use it
                    continue
                
                corner_dict[corner_info].append([x, y, i, j])
                inter_points.append([x, y])
    
    square_list = []
    connect_list = []
    segments_list = []
    for corner0 in corner_dict[0]:
        for corner1 in corner_dict[1]:
            connect01 = False
            for corner0_line in corner0[2:]:
                if corner0_line in corner1[2:]:
                    connect01 = True
                    break
            if connect01:
                for corner2 in corner_dict[2]:
                    connect12 = False
                    for corner1_line in corner1[2:]:
                        if corner1_line in corner2[2:]:
                            connect12 = True
                            break
                    if connect12:
                        for corner3 in corner_dict[3]:
                            connect23 = False
                            for corner2_line in corner2[2:]:
                                if corner2_line in corner3[2:]:
                                    connect23 = True
                                    break
                            if connect23:
                                for corner3_line in corner3[2:]:
                                    if corner3_line in corner0[2:]:
                                        # SQUARE!!!
                                        '''
                                        0 -- 1
                                        |    |
                                        3 -- 2
                                        square_list:
                                            order: 0 > 1 > 2 > 3
                                            | x0, y0, x1, y1, x2, y2, x3, y3 |
                                            | x0, y0, x1, y1, x2, y2, x3, y3 |
                                            ...
                                        connect_list:
                                            order: 01 > 12 > 23 > 30
                                            | line_idx01, line_idx12, line_idx23, line_idx30 |
                                            | line_idx01, line_idx12, line_idx23, line_idx30 |
                                            ...
                                        segments_list:
                                            order: 0 > 1 > 2 > 3
                                            | line_idx0_i, line_idx0_j, line_idx1_i, line_idx1_j, line_idx2_i, line_idx2_j, line_idx3_i, line_idx3_j |
                                            | line_idx0_i, line_idx0_j, line_idx1_i, line_idx1_j, line_idx2_i, line_idx2_j, line_idx3_i, line_idx3_j |
                                            ...
                                        '''
                                        square_list.append(corner0[:2] + corner1[:2] + corner2[:2] + corner3[:2])
                                        connect_list.append([corner0_line, corner1_line, corner2_line, corner3_line])
                                        segments_list.append(corner0[2:] + corner1[2:] + corner2[2:] + corner3[2:])
    
    def check_outside_inside(segments_info, connect_idx):
        # return 'outside or inside', min distance, cover_param, peri_param
        if connect_idx == segments_info[0]:
            check_dist_mat = dist_inter_to_segment1
        else:
            check_dist_mat = dist_inter_to_segment2
        
        i, j = segments_info
        min_dist, max_dist = check_dist_mat[i, j, :]
        connect_dist = dist_segments[connect_idx]
        if max_dist > connect_dist:
            return 'outside', min_dist, 0, 1
        else:
            return 'inside', min_dist, -1, -1


    top_square = None

    try:
        map_size = input_shape[0] / 2
        squares = np.array(square_list).reshape([-1,4,2])
        score_array = []
        connect_array = np.array(connect_list)
        segments_array = np.array(segments_list).reshape([-1,4,2])
        
        # get degree of corners:
        squares_rollup = np.roll(squares, 1, axis=1)
        squares_rolldown = np.roll(squares, -1, axis=1)
        vec1 = squares_rollup - squares
        normalized_vec1 = vec1 / (np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-10)
        vec2 = squares_rolldown - squares
        normalized_vec2 = vec2 / (np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-10)
        inner_products = np.sum(normalized_vec1 * normalized_vec2, axis=-1) # [n_squares, 4]
        squares_degree = np.arccos(inner_products) * 180 / np.pi # [n_squares, 4]

        # get square score
        overlap_scores = []
        degree_scores = []
        length_scores = []
        
        for connects, segments, square, degree in zip(connect_array, segments_array, squares, squares_degree):
            '''
            0 -- 1
            |    |
            3 -- 2
            
            # segments: [4, 2]
            # connects: [4]
            '''
            
            ###################################### OVERLAP SCORES
            cover = 0
            perimeter = 0
            # check 0 > 1 > 2 > 3
            square_length = []
             
            for start_idx in range(4):
                end_idx = (start_idx + 1) % 4
                
                connect_idx = connects[start_idx] # segment idx of segment01
                start_segments = segments[start_idx]
                end_segments = segments[end_idx]
                
                start_point = square[start_idx]
                end_point = square[end_idx]
                
                # check whether outside or inside
                start_position, start_min, start_cover_param, start_peri_param = check_outside_inside(start_segments, connect_idx)
                end_position, end_min, end_cover_param, end_peri_param = check_outside_inside(end_segments, connect_idx)
                
                cover += dist_segments[connect_idx] + start_cover_param * start_min + end_cover_param * end_min
                perimeter += dist_segments[connect_idx] + start_peri_param * start_min + end_peri_param * end_min
                
                square_length.append(dist_segments[connect_idx] + start_peri_param * start_min + end_peri_param * end_min)
            
            overlap_scores.append(cover / perimeter)    
            ######################################
            ###################################### DEGREE SCORES
            '''
            deg0 vs deg2
            deg1 vs deg3
            '''
            deg0, deg1, deg2, deg3 = degree
            deg_ratio1 = deg0 / deg2
            if deg_ratio1 > 1.0:
                deg_ratio1 = 1 / deg_ratio1
            deg_ratio2 = deg1 / deg3
            if deg_ratio2 > 1.0:
                deg_ratio2 = 1 / deg_ratio2
            degree_scores.append((deg_ratio1 + deg_ratio2) / 2)
            ######################################
            ###################################### LENGTH SCORES
            '''
            len0 vs len2
            len1 vs len3
            '''
            len0, len1, len2, len3 = square_length
            len_ratio1 = len0 / len2 if len2 > len0 else len2 / len0
            len_ratio2 = len1 / len3 if len3 > len1 else len3 / len1
            length_scores.append((len_ratio1 + len_ratio2) / 2)

            ######################################
        
        overlap_scores = np.array(overlap_scores)
        overlap_scores /= np.max(overlap_scores)
            
        degree_scores = np.array(degree_scores)
        #degree_scores /= np.max(degree_scores)
        
        length_scores = np.array(length_scores)

        ###################################### AREA SCORES
        area_scores = np.reshape(squares, [-1, 4, 2])
        area_x = area_scores[:,:,0]
        area_y = area_scores[:,:,1]
        correction = area_x[:,-1] * area_y[:,0] - area_y[:,-1] * area_x[:,0]
        area_scores = np.sum(area_x[:,:-1] * area_y[:,1:], axis=-1) - np.sum(area_y[:,:-1] * area_x[:,1:], axis=-1)
        area_scores = 0.5 * np.abs(area_scores + correction)
        area_scores /= (map_size * map_size) #np.max(area_scores)
        ######################################
        
        ###################################### CENTER SCORES
        centers = np.array([[256 // 2, 256 // 2]], dtype='float32') # [1, 2]
        # squares: [n, 4, 2]
        square_centers = np.mean(squares, axis=1) # [n, 2]
        center2center = np.sqrt(np.sum((centers - square_centers) ** 2, axis=1))
        center_scores = center2center / (map_size / np.sqrt(2.0))


        '''
        score_w = [overlap, degree, area, center, length]
        '''
        score_w = [0.0, 1.0, 10.0, 0.5, 1.0]
        score_array = params['w_overlap'] * overlap_scores \
                        + params['w_degree'] * degree_scores \
                        + params['w_area'] * area_scores \
                        - params['w_center'] * center_scores \
                        + params['w_length'] * length_scores

        best_square = []

        sorted_idx = np.argsort(score_array)[::-1]
        score_array = score_array[sorted_idx]
        squares = squares[sorted_idx]

    except Exception as e:
        pass
    
    try:
        new_segments[:,0] = new_segments[:,0] * 2 / input_shape[1] * original_shape[1]
        new_segments[:,1] = new_segments[:,1] * 2 / input_shape[0] * original_shape[0]
        new_segments[:,2] = new_segments[:,2] * 2 / input_shape[1] * original_shape[1]
        new_segments[:,3] = new_segments[:,3] * 2 / input_shape[0] * original_shape[0]
    except:
        new_segments = []
    
    try:
        squares[:,:,0] = squares[:,:,0] * 2 / input_shape[1] * original_shape[1]
        squares[:,:,1] = squares[:,:,1] * 2 / input_shape[0] * original_shape[0]
    except:
        squares = []
        score_array = []

    try:
        inter_points = np.array(inter_points)
        inter_points[:,0] = inter_points[:,0] * 2 / input_shape[1] * original_shape[1]
        inter_points[:,1] = inter_points[:,1] * 2 / input_shape[0] * original_shape[0]
    except:
        inter_points = []
        

    return new_segments, squares, score_array, inter_points

model_m_lsd = "M-LSD_320_tiny_fp16.tflite"
interpreter_m_lsd = tf.lite.Interpreter(model_path=model_m_lsd)

interpreter_m_lsd.allocate_tensors()
input_details_m_lsd = interpreter_m_lsd.get_input_details()
output_details_m_lsd = interpreter_m_lsd.get_output_details()

model_guitar_detect = YOLO("guitar_detect.pt")
threshold_guitar_detect = 0.3

model_fretboard_detect = YOLO("fretboard_detect.pt")
threshold_fretboard_detect = 0.3

model_guitar_seg = YOLO("guitar_seg.pt")
threshold_guitar_seg = 0.3

model_fretboard_seg = YOLO("fretboard_seg.pt")
threshold_guitar_seg = 0.3

img_cut_value = 0

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
tolerance = 10
dist_min = 100
dist_max = 120

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

def onChange_min_dist(val):
    global dist_min
    dist_min = val

def onChange_max_dist(val):
    global dist_max
    dist_max = val

def onChange_tolerance(val):
    global tolerance
    tolerance = val

cv2.namedWindow("Edges")
cv2.createTrackbar("lower", "Edges", canny_lower, 1000, onChange_lower)
cv2.createTrackbar("upper", "Edges", canny_upper, 1000, onChange_upper)
cv2.createTrackbar("angle", "Edges", angle_global, 90, onChange_angle)
cv2.createTrackbar("threshold", "Edges", threshold_prob, 500, onChange_threshold)
cv2.createTrackbar("tolerance", "Edges", tolerance, 100, onChange_tolerance)
cv2.createTrackbar("dist_min", "Edges", dist_min, 1000, onChange_min_dist)
cv2.createTrackbar("dist_max", "Edges", dist_max, 1000, onChange_max_dist)

while True:
    success, img = cap.read()
    if mirror_effect:
        img = cv2.flip(img, 1)
    
    img_guitar, angle_guitar = object_detect(img, model_guitar_detect)
    if img_guitar is not None and angle_guitar is not None:
        img_fretboard, angle_fretboard = object_detect(img_guitar, model_fretboard_detect)
        if img_fretboard is not None and angle_fretboard is not None:
            img_fretboard = object_segment(img_fretboard, model_fretboard_seg)
            #img_cut = canny_hough_lines(img_cut, canny_lower, canny_upper, threshold_prob, angle_global, lineLength)
            #img_cut = canny_hough_line_prob(img_fretboard, canny_lower, canny_upper, threshold_prob)
            #img_cut = canny(img_cut)
            #lines = pred_lines(img_fretboard, interpreter_m_lsd, input_details_m_lsd, output_details_m_lsd, input_shape=[320, 320], score_thr=0.1, dist_thr_min=dist_min, dist_thr_max=dist_max)
            #if lines is not None:
            #    for line in lines:
            #        x_start, y_start, x_end, y_end = [int(val) for val in line]
            #        cv2.line(img_fretboard, (x_start, y_start), (x_end, y_end), (100, 100, 100), 10)
            #lines, squares, score_array, inter_points = pred_squares(img_fretboard, interpreter_m_lsd, input_details_m_lsd, output_details_m_lsd, [320,320])
            #for square in squares:
            #    cv2.polylines(img_fretboard, [square.reshape([-1, 1, 2])], True, [200,200,0], 2)
            
            #for square in squares[0:1]:
            #    cv2.polylines(img_fretboard, [square.reshape([-1, 1, 2])], True, [255,255,0], 5)
            #    for pt in square:
            #        cv2.circle(img_fretboard, (int(pt[0]), int(pt[1])), 5, [0,255,255], -1)
    else:
        continue
    hand_proc(img=img)

    cv2.imshow("img", img)
    if img_guitar is not None:       
        cv2.imshow("img_guitar", img_guitar)
    if img_fretboard is not None:
        cv2.imshow("img_fretboard", img_fretboard)
        #cv2.imshow("img_cut", img_cut)
    
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows() 