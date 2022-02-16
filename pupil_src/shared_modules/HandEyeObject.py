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
        frame = events['frame'].img
        action = self.f_content['actions'][self.current_action]
        folder = 10

        if self.current_action < len(self.f_content['actions']) and self.folder_number < folder:

            image = events['frame'].img

            text_to_display = f'action = {action} frame_number = {self.frame_num}'  # passed_time ={passed_time}'
            cv2.putText(frame, text_to_display, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.25, (255, 100, 120), 1,
                        cv2.LINE_AA)
            mediapipe_hands = self.hands.Hands(min_tracking_confidence=0.55, min_detection_confidence=0.6)
            results = detect_hands(frame, mediapipe_hands)
            file_name = f'{self.frame_num}.npy'
            keypoints = extract_keypoints(results, events, image)
            print(keypoints)

            keypoints_path = os.path.join(self.root_folder, action, str(self.folder_number), file_name)
            np.save(keypoints_path, keypoints)
            print(keypoints)
            self.frame_num += 1

            self.draw_landmarks(image, results)
            if self.frame_num >= 20:
                # if self.frame_num >= 40:
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


def extract_keypoints(results, occurrence, image):
    if results.multi_hand_landmarks:
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

                fixation_pts = np.array([pt["norm_pos"] for pt in occurrence.get(
                    "fixations")]).flatten() if occurrence.get('fixations') else np.zeros(2)
                bbox = extract_bbox(image)
                return np.concatenate([left_hand, right_hand, fixation_pts, bbox])


def extract_bbox(results, empty=np.zeros(4)):
    if not results:
        return empty
    for i in results:
        if i['class'] == 3:
            empty[0] = i['xcenter']
            empty[1] = i['ycenter']
            empty[2] = i['width']
            empty[3] = i['height']
            return empty.flatten()
        return empty
