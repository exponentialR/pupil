import cv2
import glfw
import mediapipe as mp
import numpy as np
from pyglui import ui
from pyglui.cygl.utils import draw_polyline, draw_points, RGBA, draw_gl_texture

import gl_utils
from threading import Thread
from multiprocessing import Process, Queue, Pool

mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from plugin import Plugin
# logging
import logging

logger = logging.getLogger(__name__)


class Hand_Pupil(Plugin):
    icon_chr = "HPC"
    icon_font = "roboto"
    """Describe your plugin here
    """

    def __init__(self, g_pool, frame=None, run=False):
        super(Hand_Pupil, self).__init__(g_pool)
        queue_size = 1
        self.queue_size = queue_size
        self.input_h = Queue(maxsize=self.queue_size)
        self.output_h = Queue(maxsize=self.queue_size)
        self.g_pool.display_mode = "algorithm"
        self.frame = frame

        self.order = .1
        self.queue_size = queue_size
        self.hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4)
        self.run = run
        self.window = None
        self.menu = None
        self.img = None
        # Thread

        p = Thread(target=self.worker, args=(self.input_h, self.output_h))
        p.daemon = True
        p.start()

        self.activation_state = False
        self.new_window = False

    def recent_events(self, events):
        if 'frame' in events:
            frame = events['frame']

            self.img = frame.img
            self.frame = self.img
            self.input_h.put(self.img)
            m_results = self.output_h.get()
            if not m_results.multi_hand_landmarks:
                return
            self.Drawlandmark(self.input_h.get(), m_results)
        return self.input_h.get()

    def handDetection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return results

    def worker(self, input_h, output_h):
        while self.hands:
            frame = input_h.get()
            output_h.put(self.handDetection(frame, self.hands))

    def init_ui(self):
        # lets make a menu entry in the sidebar
        self.add_menu()
        self.menu.label = 'Hand Tracking'
        self.menu.append(
            ui.Info_Text(
                "Tracks the users hand in the world view camera"))
        # self.menu.append(ui.Switch("new_window", self, label="Open Hand Tracking in new window"))

    def on_resize(self, window, w, h):
        active_window = glfw.get_current_context()
        glfw.make_context_current(window)
        gl_utils.adjust_gl_view(w, h)
        glfw.make_context_current(active_window)

    def deinit_ui(self):
        self.remove_menu()

    def Drawlandmark(self, image, m_results):
        for hand_landmarks in m_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)



            # with mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4) as hands:
            #     m_results = self.handDetection(self.img, hands)
            #
            #     if not m_results.multi_hand_landmarks:
            #         return
            #     self.Drawlandmark(self.img, m_results)
