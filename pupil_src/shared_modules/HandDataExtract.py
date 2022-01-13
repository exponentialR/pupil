import os.path
import os

import cv2
import mediapipe as mp
import numpy as np
# import gl_utils

from pathlib import Path
from time import perf_counter
import json


current_working_directory = os.path.dirname(__file__)
relative_data_path = 'data/iamsMediapipe.iams'
absolute_iams_path = Path(os.path.join(current_working_directory, relative_data_path))
ROOT_FOLDER = os.path.join(current_working_directory, 'data')
f = open(absolute_iams_path)
# f_content = json.load(f)
time_per_action_sec = 2.0
# root_folder = "data/handdata"  # or self.f_content['subfolder_directory']

from plugin import Plugin
import logging

logger = logging.getLogger((__name__))


class HandDataExtract(Plugin):
    """"This Plugin Extracts hand coordinates
    and fixation in world scene camera"""
    icon_chr = "HP"
    icon_font = "roboto"

    def __init__(self, g_pool):
        super(HandDataExtract, self).__init__(g_pool)
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
        start_ = perf_counter()
        if self.start_time is None:
            self.start_time = start_
        passed_time = start_ - self.start_time
        action = self.f_content["actions"][self.current_action]
        sequence = self.sequence
        if passed_time <= time_per_action_sec and self.frame_num <=40:

            self.passed_time.append(passed_time)
            frame = events['frame'].img
            # y = frame.get(5)
            # print(y)
            text_in_image = f"{action} idx={self.frame_num} passed_time={passed_time}"
            cv2.putText(frame, text_in_image, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1, cv2.LINE_AA)
            with self.hands.Hands(min_tracking_confidence=0.55, min_detection_confidence=0.6) as inference_mod:
                results, image = self.detect_hands(frame, inference_mod)
                keypoints_fixations = self.extract_coordinates(results, events)
                file_name = f"keypoint_{self.frame_num}"
                self.frame_num += 1
                path_to_keypoint_file = Path(os.path.join(self.root_folder, action, str(sequence), file_name))

                np.save(path_to_keypoint_file, keypoints_fixations)

                print(keypoints_fixations)
                if results.multi_hand_landmarks:
                    # self.draw_landmarks(frame, results)
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.hands.HAND_CONNECTIONS)

                    # self.frame_num += 1
                    # self.sequence +=1
                    # os.path.join(self.f_content['Data_Subfolder'], self.f_content['actions'][0], str(self.frame_num))
                    # for sequence in range(self.sequence):



                    if sequence >=5:
                        self.current_action += 1
                        time_file_name = "time_since_action_start.npy"
                        path_to_time_file = os.path.join(self.root_folder, action, time_file_name)
                        np.save(path_to_time_file, self.passed_time)
                        self.passed_time = []

                        self.start_time = None
                        # sequence +=1
                        # self.current_action += 1
                        # cv2.waitKey(5000)
                        if self.current_action > len(self.f_content["actions"]):
                            self.running = False
                        return


        else:
            time_file_name = "time_since_action_start.npy"
            path_to_time_file = os.path.join(self.root_folder, action, str(sequence), time_file_name)
            np.save(path_to_time_file, self.passed_time)
            self.passed_time = []
            self.start_time = None
            self.sequence +=1
            self.frame_num = 1

            if self.sequence <= sequence:
                return


