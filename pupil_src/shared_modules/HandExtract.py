# import concurrent.futures
# import itertools
import concurrent.futures
import cv2
import glfw
import mediapipe as mp
import numpy as np
from pyglui import ui
import os
from pyglui.cygl.utils import draw_polyline, draw_points, RGBA, draw_gl_texture

import gl_utils
# from threading import Thread
# from multiprocessing import Process
from multiprocessing import Queue


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

from plugin import Plugin
# logging
import logging

logger = logging.getLogger(__name__)



def make_sequence_folder():
    list_action = ['Intention', 'No_intention']
    DATA_PATH = 'hand_pupil_data'
    # y = DATA_PATH
    sequence_length = 30
    no_sequences = 30
    i = 1
    if not os.path.exists(DATA_PATH):
        DATA_PATH = os.path.join(DATA_PATH)

    while os.path.exists(DATA_PATH):
        DATA_PATH = ('hand_pupil_data_%s' % i)
        i += 1
        DATA_PATH = os.path.join(DATA_PATH)

    for action in list_action:
        # dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        for sequence in range(0, no_sequences + 1):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    return list_action, DATA_PATH, sequence_length, no_sequences


# queue = multiprocessing.Queue

class HandExtract(Plugin):
    icon_chr = "HP"
    icon_font = "roboto"
    """Describe youself.input = multiprocessing.Queue()r plugin here
    """

    def __init__(self, g_pool):
        super(HandExtract, self).__init__(g_pool)
        self.m = np.zeros(42 * 3)
        self.g_pool.display_mode = "algorithm"
        self.order = .01
        self.window = None
        self.menu = None
        self.img = None
        self.new_window = False
        # self.input_q = Queue(maxsize=self.queue_size)
        # self.output_q = Queue(maxsize=self.queue_size)
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.65)
        self.Draw = mp_drawing.draw_landmarks
        self.states, self.pathdir, self.seq_len, self.seq_no = make_sequence_folder()

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Hand Tracking'
        self.menu.append(
            ui.Info_Text(
                "Tracks the users hand in the world view camera"))

    def on_resize(self, window, w, h):
        active_window = glfw.get_current_context()
        glfw.make_context_current(window)
        gl_utils.adjust_gl_view(w, h)
        glfw.make_context_current(active_window)

    def deinit_ui(self):
        self.remove_menu()

    def handDetection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        model = self.holistic.process(image)
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # output.put(model)

    def draw_keypoints(self, input1, output):
        self.Draw(input1, output.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        self.Draw(input1, output.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def extract_key(self, results):
        lh = np.array([[res.x, res.y, res.z] for res in
                       results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in
                       results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([lh, rh])

    def recent_events(self, events):
        if "frame" not in events:
            return
        frame = events['frame']
        self.img = frame.img

        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)

        image.flags.writeable = True
        self.Draw(self.img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        self.Draw(self.img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        keypoints = self.extract_key(results)
        print(keypoints)

