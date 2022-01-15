import json
import os
import os.path
from pathlib import Path

import mediapipe as mp
import numpy as np

# import gl_utils

current_working_directory = os.path.dirname(__file__)
relative_data_path = 'data/iamsMediapipe.iams'
absolute_iams_path = Path(os.path.join(current_working_directory, relative_data_path))
ROOT_FOLDER = os.path.join(current_working_directory, 'data')
f = open(absolute_iams_path)
# f_content = json.load(f)
time_per_action_sec = 2.0
# root_folder = "data/handdata"  # or self.f_content['subfolder_directory']
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = torch.device('cuda')
path = r'/home/iamshri/PycharmProjects/intent/yolov5/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path)
model.to(device)

from plugin import Plugin
import logging

logger = logging.getLogger((__name__))


class ObjectDetector(Plugin):
    """"This Plugin Extracts hand coordinates
    and fixation in world scene camera"""
    icon_chr = "HP"
    icon_font = "roboto"

    def __init__(self, g_pool):
        super(ObjectDetector, self).__init__(g_pool)
        self.g_pool.display_mode = "algorithm"
        self.order = 1.0
        self.window = None
        self.menu = None
        self.img = None
        self.new_window = False
        self.f_content = json.load(f)
        # self.root_folder = self.f_content['Data_Subfolder']
        self.root_folder = os.path.join(ROOT_FOLDER, self.f_content['Data_Directory'])
        self.frame_num = 1
        self.current_action = 0
        self.start_time = None
        self.passed_time = []
        self.running = True
        self.hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        self.sequence = 0
        self.path = r'/home/iamshri/PycharmProjects/intent/yolov5/best.pt'

    @staticmethod
    def detect_hands(image, result):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = result.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image

    def draw_landmarks(self, image, results):
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(image, hand_landmarks, self.hands.HAND_CONNECTIONS)

    # def leftorright(self, image, coords):
    #     for idx, hand_landmarks in enumerate(coords.multi_hand_landmarks):
    #         self.mp_drawing.draw_landmarks(image, hand_landmarks, self.hands.HAND_CONNECTIONS, self.mp_drawing_styles.get_default_hand_landmarks_style(), self.mp_drawing_styles.get_default_hand_connections_style)
    #     for _, classification in enumerate(coords.multi_handedness):
    #         if classification.classification[0].index == idx and classification

    @staticmethod
    def extract_coordinates(results, occurrence):
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
                        "fixations")]).flatten() if "fixations" in occurrence else np.zeros(2 * 2)
                    # fixation_pts = [pt["norm_pos"] for pt in events.get("fixations")]
                    # fixation_pts = np.array([pt["norm_pos"] for pt in events.get("fixations")]).flatten() if "fixations" in events else np.zeros(2*2)
                    return np.concatenate([fixation_pts, right_hand, left_hand])

    def extract_fixations(self, occurrence):
        return np.array([pt["norm_pos"] for pt in occurrence.get(
            "fixations")]).flatten() if "fixations" in occurrence else np.zeros(2 * 2)

    def recent_events(self, events):
        if not self.running or "frame" not in events:
            return
        frame = events['frame'].img
        detect_obj = model(frame[..., ::-1])
        results = detect_obj.pandas().xyxy[0].to_dict(orient='records')
        print(results)
            # if results.multi_hand_landmarks:
            # for idx, hand_coordinates in enumerate(results.multi_hand_landmarks):
            #     for _, classification in enumerate(results.multi_handedness):
            #         right_hand = np.array([[coordinates_.x, coordinates_.y, coordinates_.z] for coordinates_ in
            #                                hand_coordinates.landmark]).flatten() if classification.classification[
            #                                                                             0].index == 1 else np.zeros(
            #             21 * 3)
            #         left_hand = np.array([[coordinates_.x, coordinates_.y, coordinates_.z] for coordinates_ in
            #                               hand_coordinates.landmark]).flatten() if classification.classification[
            #                                                                            0].index == idx else np.zeros(
            #             21 * 3)
        for results_ in results:
            results__ = np.array([[int(results_['xmin']), int(results_['ymin']), int(results_['xmax']), int(results_['ymax'])] for results_ in results]).flatten()  if results_['class'] == 1 else np.zeros(4)
            rec_coords = np.array(results__)
            print(rec_coords)
            x1, y1, x2, y2 = rec_coords[0], rec_coords[1], rec_coords[2], rec_coords[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
