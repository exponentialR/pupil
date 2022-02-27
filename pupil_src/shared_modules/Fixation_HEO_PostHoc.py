from plugin import Plugin
import json
import cv2
from pathlib import Path
import numpy as np
import os

current_working_directory = os.path.dirname(__file__)
relative_data_path = 'data/iamsMediapipe.iams'
absolute_iams_path = Path(os.path.join(current_working_directory, relative_data_path))
ROOT_FOLDER = os.path.join(current_working_directory, 'data')
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = torch.device('cuda')
path = r'/home/iamshri/PycharmProjects/intent/yolov5/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path)
model.to(device)


class Fixation_HEO_PostHoc(Plugin):
    """"This Plugin Extracts hand coordinates
    and fixation in world scene camera"""
    icon_chr = "HP"
    icon_font = "roboto"

    def __init__(self, g_pool):
        super(Fixation_HEO_PostHoc, self).__init__(g_pool)
        self.g_pool.display_mode = "algorithm"
        self.order = 1.0
        self.f = open(absolute_iams_path, 'r')
        self.new_window = False
        self.f_content = json.load(self.f)
        self.root_folder = os.path.join(ROOT_FOLDER, self.f_content['Data_Directory'])
        self.frame_num = 1
        self.current_action = self.f_content['action_number']
        self.passed_time = []
        self.running = True
        self.folder_number = 1
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_path = os.path.join(self.root_folder, self.f_content['actions'][self.current_action])
        self.output_video = cv2.VideoWriter('{}.avi'.format(self.video_path), self.fourcc, 20.0, (640, 480))


    def recent_events(self, events):
        if not self.running or 'frame' not in events:
            return
        frame = events['frame'].img
        fixation_pts = np.array([pt["norm_pos"] for pt in events.get("fixations")]).flatten() if events.get(
            'fixations') else np.zeros(2)
        action = self.f_content['actions'][self.current_action]

        detect_obj = model(events['frame'].img[..., ::-1])
        results_obj = np.array(detect_obj.pandas().xywhn[0].to_dict(orient='records'))
        bbox = extract_bbox(results_obj, empty=np.zeros(4))
        plot_bbox = np.array(detect_obj.pandas().xyxy[0].to_dict(orient='records'))
        draw_plots(frame, plot_bbox)

        while self.running:
            if self.folder_number <= 30:
                text_to_display = 'current action : {} current folder : {}, frame number : {} '.format(action, self.folder_number, self.frame_num)
                cv2.putText(events['frame'].img, text_to_display, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.25, (255, 100, 250), 1, cv2.LINE_AA)
                file_name = f'{self.frame_num}.npy'
                Eye_Object_keypoints = np.concatenate([fixation_pts, bbox])
                Eye_Object_keypoints_path = os.path.join(self.root_folder, action, str(self.folder_number), file_name)
                np.save(Eye_Object_keypoints_path, Eye_Object_keypoints)

                # print(events['frame'].img.shape)

                self.frame_num +=1
                self.output_video.write(frame)

                if self.frame_num >= 60:
                    self.folder_number += 1
                    self.frame_num = 1

            else:
                self.current_action += 1
                if self.current_action <= len(self.f_content['actions']):
                    with open(absolute_iams_path, 'r') as f:
                        data = json.load(f)
                    data['action_number'] = self.current_action
                    with open(absolute_iams_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    self.running = False
                    return
                self.running = False
                return
            return


def draw_plots(frame, results):
    for res in results:  # plot bounding boxes and include labels
        if res['class'] == 0:  # Book
            le = int(res['xmin'])
            t = int(res['ymin'])
            r = int(res['xmax'])
            b = int(res['ymax'])
            text_in_image = res['name']
            cv2.rectangle(frame, (le, t), (r, b), (0, 0, 255), 1)
            cv2.putText(frame, text_in_image, (le, t), cv2.FONT_HERSHEY_DUPLEX, 0.35, (255, 100, 120), 1,
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
            pass
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
