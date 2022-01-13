import time

import cv2
import numpy as np
import os
from pathlib import Path
from time import perf_counter
import json

current_working_directory = os.path.dirname(__file__)
relative_data_path = 'data/iamsfixation.iams'
absolute_iams_path = Path(os.path.join(current_working_directory, relative_data_path))
ROOT_FOLDER = os.path.join(current_working_directory, 'data')
f = open(absolute_iams_path)
# f_content = json.load(f)
time_per_action_sec = 30.0

from plugin import Plugin
import logging

logger = logging.getLogger((__name__))


class Fixation_Extractor(Plugin):
    """"This Plugin Extracts eye fixation coordinates to a numpy array
    and fixation in world scene camera"""
    icon_chr = "HP"
    icon_font = "roboto"

    def __init__(self, g_pool):
        super(Fixation_Extractor, self).__init__(g_pool)
        self.g_pool.display_mode = "algorithm"
        self.order = .5
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

    @staticmethod
    def extract_fixations(occurrence):
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

        if passed_time <= time_per_action_sec:
            self.passed_time.append(passed_time)
            frame = events['frame'].img
            text_in_image = f"{action} idx={self.frame_num} passed_time={passed_time}"

            cv2.putText(frame, text_in_image, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.waitKey(7000)
            fixation_extract = self.extract_fixations(events)
            file_name = f"keypoint_{self.frame_num}"
            # os.path.join(self.f_content['Data_Subfolder'], self.f_content['actions'][0], str(self.frame_num))
            path_to_keypoint_file = Path(os.path.join(self.root_folder, action, file_name))
            np.save(path_to_keypoint_file, fixation_extract)
            print(fixation_extract)

        else:
            # cv2.waitKey(7000)
            time_file_name = "time_since_action_start.npy"
            path_to_time_file = os.path.join(self.root_folder, action, time_file_name)
            np.save(path_to_time_file, self.passed_time)
            self.passed_time = []
            # self.frame_num = 1
            self.start_time = None
            self.current_action += 1
            # cv2.waitKey(5000)
            if self.current_action > len(self.f_content["actions"]):
                self.running = False
            return

