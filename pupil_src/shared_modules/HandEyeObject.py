import os
import time
from plugin import Plugin
import json
import cv2
from pathlib import Path
import numpy as np

current_working_directory = os.path.dirname(__file__)
relative_data_path = 'data/iamsMediapipe.iams'
absolute_iams_path = Path(os.path.join(current_working_directory, relative_data_path))
ROOT_FOLDER = os.path.join(current_working_directory, 'data')
time_per_action_sec = 30.0
root_folder = "data/handdata"  # or self.f_content['subfolder_directory']

import mediapipe as mp
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = torch.device('cuda')
path = r'/home/iamshri/PycharmProjects/intent/yolov5/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path)
model.to(device)


def detect_hands(image, result):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = result.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results


def extract_keypoints(results):
    if not results.multi_hand_landmarks:
        return np.concatenate([np.zeros(63), np.zeros(63)])
    for idx, hand_coordinates in enumerate(results.multi_hand_landmarks):
        for _, classification in enumerate(results.multi_handedness):
            right_hand = np.array([[coordinates_.x, coordinates_.y, coordinates_.z] for coordinates_ in
                                   hand_coordinates.landmark]).flatten() if classification.classification[
                                                                                0].index == 1 else np.zeros(
                21 * 3)
            left_hand = np.array([[coordinates_.x, coordinates_.y, coordinates_.z] for coordinates_ in
                                  hand_coordinates.landmark]).flatten() if classification.classification[
                                                                               0].index == idx else np.zeros(
                21 * 3)
            return np.concatenate([left_hand, right_hand])


class HandEyeObject(Plugin):
    """"This Plugin Extracts hand coordinates
    and fixation in world scene camera"""
    icon_chr = "HP"
    icon_font = "roboto"

    def __init__(self, g_pool):
        super(HandEyeObject, self).__init__(g_pool)
        self.g_pool.display_mode = "algorithm"
        self.order = 1.0
        self.f = open(absolute_iams_path)
        self.new_window = False
        self.f_content = json.load(self.f)
        # self.root_folder = self.f_content['Data_Subfolder']
        self.root_folder = os.path.join(ROOT_FOLDER, self.f_content['Data_Directory'])
        self.frame_num = 1
        self.current_action = 0
        self.start_time = None
        self.passed_time = []
        self.running = True
        self.hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.folder_number = 0

    def recent_events(self, events):
        if not self.running or 'frame' not in events:
            return
        fixation_pts = np.array([pt["norm_pos"] for pt in events.get(
            "fixations")]).flatten() if events.get('fixations') else np.zeros(2)
        action = self.f_content['actions'][self.current_action]
        folder = 30

        detect_obj = model(events['frame'].img[..., ::-1])
        results_obj = np.array(detect_obj.pandas().xywhn[0].to_dict(orient='records'))
        # plot_bbox = np.array(detect_obj.pandas().xyxy[0].to_dict(orient='records'))
        bbox = extract_bbox(results_obj, empty=np.zeros(4))
        # draw_plots(events['frame'].img, plot_bbox)
        with self.hands.Hands(min_tracking_confidence=0.55, min_detection_confidence=0.6) as mediapipe_hands:

            if self.current_action < len(self.f_content['actions']) and self.folder_number < folder:
                image = events['frame'].img
                text_to_display = 'current action : {} current folder : {}, frame number : {} '.format(action, self.folder_number, self.frame_num)
                # f'action = {action} frame_number = {self.frame_num}'  # passed_time ={passed_time}'
                cv2.putText(events['frame'].img, text_to_display, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.25,
                            (255, 100, 250), 1,
                            cv2.LINE_AA)
                results = detect_hands(events['frame'].img, mediapipe_hands)
                self.draw_landmarks(image, results)

                file_name = f'{self.frame_num}.npy'
                hand_keypoints = extract_keypoints(results)
                H_E_O_keypoints = np.concatenate([hand_keypoints, fixation_pts, bbox])
                H_E_O_keypoints_path = os.path.join(self.root_folder, action, str(self.folder_number), file_name)
                np.save(H_E_O_keypoints_path, H_E_O_keypoints)
                self.frame_num += 1

                if self.frame_num >= 60:
                    self.folder_number += 1
                    self.frame_num = 1

            else:
                self.current_action += 1
                self.folder_number = 0
                if self.current_action > len(self.f_content['actions']):
                    self.running = False

                return

    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.hands.HAND_CONNECTIONS)


def draw_plots(frame, results):
    for res in results:  # plot bounding boxes and include labels
        if res['class'] == 0:  # Book
            l = int(res['xmin'])
            t = int(res['ymin'])
            r = int(res['xmax'])
            b = int(res['ymax'])
            text_in_image = res['name']
            cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)
            cv2.putText(frame, text_in_image, (l, t), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
                        cv2.LINE_AA)

        elif res['class'] == 1:  # Mug
            l1 = int(res['xmin'])
            t1 = int(res['ymin'])
            r1 = int(res['xmax'])
            b1 = int(res['ymax'])
            text_in_image = res['name']
            cv2.rectangle(frame, (l1, t1), (r1, b1), (255, 0, 0), 1)
            cv2.putText(frame, text_in_image, (l1, t1), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
                        cv2.LINE_AA)

        elif res['class'] == 2:  # Mugs
            l2 = int(res['xmin'])
            t2 = int(res['ymin'])
            r2 = int(res['xmax'])
            b2 = int(res['ymax'])
            text_in_image = res['name']
            # cv2.rectangle(frame, (l2, t2), (r2, b2), (255, 0, 255), 1)
            # cv2.putText(frame, text_in_image, (l2, t2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
            #             cv2.LINE_AA)

        else:
            l3 = int(res['xmin'])
            t3 = int(res['ymin'])
            r3 = int(res['xmax'])
            b3 = int(res['ymax'])
            text_in_image = res['name']
            cv2.rectangle(frame, (l3, t3), (r3, b3), (0, 255, 0), 1)
            cv2.putText(frame, text_in_image, (l3, t3), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
                        cv2.LINE_AA)


def extract_bbox(results, empty=np.zeros(4)):
    for i in results:
        if i['class'] == 3:
            empty[0] = i['xcenter']
            empty[1] = i['ycenter']
            empty[2] = i['width']
            empty[3] = i['height']
            return empty.flatten()
        return empty
    return np.zeros(4)

#try this code Monday 21.2.22
def extract_obj(results, empty=np.zeros(4)):
    if not results:
        return empty.flatten()
    while results['class'] == 3:
        empty[0] = results['xcenter']
        empty[1] = results['ycenter']
        empty[2] = results['width']
        empty[3] = results['height']
        return empty.flatten()
    else:
        return empty.flatten()


# def extract_obb(results, empty=np.zeros(4)):
#     if not results['class'] == 3:
#         return empty.flatten()
#     empty[0] = results['xcenter']
#     empty[1] = results['ycenter']
#     empty[2] = results['width']
#     empty[3] = results['height']
#     return empty.flatten()
