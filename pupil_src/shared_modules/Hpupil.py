import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from plugin import Plugin

import cv2
import glfw
import mediapipe as mp
from pyglui import ui
import gl_utils
import os
import sys
from pathlib import Path
import cv2


class Hand_Pupil(Plugin):
    icon_chr = "HPC"
    icon_font = "roboto"
    """Describe your plugin here
    """

    def __init__(self, g_pool):
        super(Hand_Pupil, self).__init__(g_pool)
        self.g_pool.display_mode = "algorithm"  # order (0-1) determines if your plugin should run before other plugins or after
        self.order = .01
        self.window = None
        self.menu = None
        self.img = None
        self.new_window = False
        self.save_txt = True
        self.view_img = True
        self.hide_labels = False
        self.img_size = 640
        self.stride = 32
        self.save_crop = False
        self.auto = True
        self.hide_conf = False

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

    def handDetection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return results

    def Drawlandmark(self, image, results):
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS)

    def recent_events(self, events):
        if "frame" not in events:
            return
        frame = events['frame'].img

        cv2.putText(frame, 'starting collection', (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 1, cv2.LINE_AA)
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            m_results = self.handDetection(frame, hands)
            if not m_results.multi_hand_landmarks:
                return
            self.Drawlandmark(frame, m_results)
