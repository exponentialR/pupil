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
names = model.names

from plugin import Plugin
import logging

logger = logging.getLogger((__name__))


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


def extract_coordinates(results):
    empty_array = np.zeros(12)
    for i in results:

        if i['class'] == 0:  # Book
            empty_array[0] = i['xcenter']
            empty_array[1] = i['ycenter']
            empty_array[2] = i['width']
            empty_array[3] = i['height']

        elif i['class'] == 1:  # Mug
            empty_array[4] = i['xcenter']
            empty_array[5] = i['ycenter']
            empty_array[6] = i['width']
            empty_array[7] = i['height']

        elif i['class'] == 3:  # Stacked Books
            empty_array[8] = i['xcenter']
            empty_array[9] = i['ycenter']
            empty_array[10] = i['width']
            empty_array[11] = i['height']
        pass

        # else:  # Mugs
        # empty_array[12] = i['xcenter']
        # empty_array[13] = i['ycenter']
        # empty_array[14] = i['width']
        # empty_array[15] = i['height']
        # print(names[1])
    return empty_array


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
        # self.f_content = json.load(f)
        # self.root_folder = self.f_content['Data_Subfolder']
        # self.root_folder = os.path.join(ROOT_FOLDER, self.f_content['Data_Directory'])
        self.frame_num = 1
        self.current_action = 0
        self.start_time = None
        self.passed_time = []
        self.running = True
        self.rect_array = np.zeros(16)
        self.hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        self.sequence = 0
        self.path = r'/home/iamshri/PycharmProjects/intent/yolov5/best.pt'

    def recent_events(self, events):
        if not self.running or "frame" not in events:
            return
        frame = events['frame'].img
        image = frame.copy()
        detect_obj = model(image[..., ::-1])
        results = detect_obj.pandas().xywhn[0].to_dict(orient='records')
        # empty_array = np.zeros(12)
        rect_bbox = detect_obj.pandas().xyxy[0].to_dict(orient='records')
        # rect_coords = extract_coordinates(results)
        # print(rect_bbox)
        draw_plots(frame, rect_bbox)
        print(self.extract_bbox(results, empty=np.zeros(4)))

    def extract_bbox(self, results, empty=np.zeros(4)):
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

        # for res in rect_bbox:  # plot bounding boxes and include labels
        #     if res['class'] == 0:  # Book
        #         l = int(res['xmin'])
        #         t = int(res['ymin'])
        #         r = int(res['xmax'])
        #         b = int(res['ymax'])
        #         text_in_image = res['name']
        #         cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)
        #         cv2.putText(frame, text_in_image, (l, t), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
        #                     cv2.LINE_AA)
        #
        #     elif res['class'] == 1:  # Mug
        #         l1 = int(res['xmin'])
        #         t1 = int(res['ymin'])
        #         r1 = int(res['xmax'])
        #         b1 = int(res['ymax'])
        #         text_in_image = res['name']
        #         cv2.rectangle(frame, (l1, t1), (r1, b1), (255, 0, 0), 1)
        #         cv2.putText(frame, text_in_image, (l1, t1), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
        #                     cv2.LINE_AA)
        #
        #     elif res['class'] == 2:  # Mugs
        #         l2 = int(res['xmin'])
        #         t2 = int(res['ymin'])
        #         r2 = int(res['xmax'])
        #         b2 = int(res['ymax'])
        #         text_in_image = res['name']
        #         # cv2.rectangle(frame, (l2, t2), (r2, b2), (255, 0, 255), 1)
        #         # cv2.putText(frame, text_in_image, (l2, t2), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
        #         #             cv2.LINE_AA)
        #
        #     else:
        #         l3 = int(res['xmin'])
        #         t3 = int(res['ymin'])
        #         r3 = int(res['xmax'])
        #         b3 = int(res['ymax'])
        #         text_in_image = res['name']
        #         cv2.rectangle(frame, (l3, t3), (r3, b3), (0, 255, 0), 1)
        #         cv2.putText(frame, text_in_image, (l3, t3), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
        #                     cv2.LINE_AA)
