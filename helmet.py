import imageio
from PIL import Image
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import subprocess

import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline
plt.rcParams['figure.dpi'] = 150

import seaborn as sns
from ipywidgets import Video

from IPython.display import display
#block those warnings from pandas about setting values on a slice
import warnings
warnings.filterwarnings('ignore')

# Read in the video labels file
video_labels = pd.read_csv('train_labels.csv')
video_labels.head()

from ipywidgets import Video
Video.from_file('57583_000082_Endzone.mp4', width=500, height=500)


# Create a function to annotate the video at the provided path using labels from the provided dataframe, return the path of the video
def annotate_video(video_path: str, video_labels: pd.DataFrame, blind=False, speed=1) -> str:
    VIDEO_CODEC = "MP4V"
    HELMET_COLOR = (255, 255, 255)    # Black
    IMPACT_COLOR = (0, 0, 255)  # Red
    video_name = os.path.basename(video_path)
    
    vidcap = cv2.VideoCapture(video_path)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "labeled_" + video_name
    tmp_output_path = "test_tmp_" + output_path
    fps = 60 * speed
    output_video = cv2.VideoWriter(tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        if blind:
            img = img * 0
        
        # We need to add 1 to the frame count to match the label frame index that starts at 1
        frame += 1
        
        # Let's add a frame index to the video so we can track where we are
        img_name = f"{video_name}_frame{frame}"

        cv2.putText(img, img_name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, HELMET_COLOR, thickness=2)
    
        # Now, add the boxes
        boxes = video_labels.query("video == @video_name and frame == @frame")
        for box in boxes.itertuples(index=False):
            if box.impact == 1 and box.confidence > 1 and box.visibility > 0:    # Filter for definitive head impacts and turn labels red
                color, thickness = IMPACT_COLOR, 2
            else:
                color, thickness = HELMET_COLOR, 1
            # Add a box around the helmet
            cv2.rectangle(img, (box.left, box.top), (box.left + box.width, box.top + box.height), color, thickness=thickness)
            cv2.putText(img, box.label, (box.left, max(0, box.top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
        output_video.write(img)
    output_video.release()
    
    
    return output_path
    
    
    labeled_video = annotate_video('57583_000082_Endzone.mp4', video_labels, speed=0.5)
    
h27 = video_labels.loc[(video_labels.label == 'H27') & (video_labels.view == 'Endzone') & (video_labels.video == '57583_000082_Endzone.mp4')]
#indexNames = h90[(h90['frame'] == 223) & (h90['frame'] == 224) & (h90['frame'] == 225) & (h90['frame'] == 226)].index
#h90.drop(indexNames , inplace=True)
h27


def newcrop_video(video_path: str, blind=False, speed=1) -> str:
    VIDEO_CODEC = "MP4V"
    video_name = os.path.basename(video_path)
    
    vidcap = cv2.VideoCapture(video_path)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "labeled_" + video_name
    tmp_output_path = "+new_crop_tmp_" + output_path
    fps = 60 * speed

    frame = 0
    
    x = h27.loc[h27['frame']==1,'left'].squeeze()-100
    y = h27.loc[h27['frame']==1,'top'].squeeze()-10
    w = h27.loc[h27['frame']==1,'width'].squeeze()+200
    h =h27.loc[h27['frame']==1,'height'].squeeze()+100

    output_video = cv2.VideoWriter(tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (w, h))
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        if blind:
            img = img * 0
        
        # We need to add 1 to the frame count to match the label frame index that starts at 1
        # w = 19+200
        # h = 24+100
        
        # Let's add a frame index to the video so we can track where we are
        frame += 1

        if frame == 403:
            break
        
        _w = h27.loc[h27['frame']==frame,'width'].squeeze()+200
        _h =h27.loc[h27['frame']==frame,'height'].squeeze()+100
        _x = h27.loc[h27['frame']==frame,'left'].squeeze()-100
        _y = h27.loc[h27['frame']==frame,'top'].squeeze()-10

        img = img[_y:_y+h, _x:_x+w]
        #print(_w,_h,frame)

        #cv2.putText(img, img_name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, HELMET_COLOR, thickness=2)
    
        # Now, add the boxes
        output_video.write(img)
    output_video.release()
    print(width,height)
    
    
    return output_path
    
croped_video = newcrop_video('57583_000082_Endzone.mp4', speed=0.5)


import mediapipe as mp
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import importlib
import nb_helpers

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
#poselandmarks_list = nb_helpers.poselandmarks_list

poselandmarks_list = []
for idx, elt in enumerate(mp_pose.PoseLandmark):
    lm_str = repr(elt).split('.')[1].split(':')[0]
    poselandmarks_list.append(lm_str)

#nb_helpers = importlib.import_module("nb_helpers")


file = '+new_crop_tmp_labeled_57583_000082_Endzone.mp4'
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    cap = cv2.VideoCapture(file)

    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    data = np.zeros((3, len(mp_pose.PoseLandmark), length))    
    

    frame_num = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        check = results.pose_world_landmarks

        if results.pose_world_landmarks == None:
            break
        
        landmarks = results.pose_world_landmarks.landmark
        for i in range(len(mp_pose.PoseLandmark)):
            #print(results.pose_world_landmarks.landmark)
            data[:, i, frame_num] = (landmarks[i].x, landmarks[i].y, landmarks[i].z)  
        
        frame_num += 1
                
    cap.release()

pose_connections = mp.solutions.pose.POSE_CONNECTIONS
mp_pose = mp.solutions.pose

poselandmarks_list = []
for idx, elt in enumerate(mp_pose.PoseLandmark):
    lm_str = repr(elt).split('.')[1].split(':')[0]
    poselandmarks_list.append(lm_str)


def scale_axes(ax):
    # Scale axes properly
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])
    
    
def time_animate(data, figure, ax, rotate_data=True, rotate_animation=False):
    frame_data = data[:, :, 0]
    if rotate_data:
        plot = [ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:blue')]

        for i in pose_connections:
            plot.append(ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                                  [-frame_data[2, i[0]], -frame_data[2, i[1]]],
                                  [-frame_data[1, i[0]], -frame_data[1, i[1]]],
                                  color='k', lw=1)[0])

        ax.view_init(elev=10, azim=120)

    else:
        ax.scatter(frame_data[0, :], frame_data[1, :], frame_data[2, :], color='tab:blue')

        for i in pose_connections:
            ax.plot3D([frame_data[0, i[0]], frame_data[0, i[1]]],
                      [frame_data[1, i[0]], frame_data[1, i[1]]],
                      [frame_data[2, i[0]], frame_data[2, i[1]]],
                      color='k', lw=1)

        ax.view_init(elev=-90, azim=-90)

    scale_axes(ax)

    def init():
        return figure,

    def animate(i):
        frame_data = data[:, :, i]

        for idxx in range(len(plot)):
            plot[idxx].remove()

        plot[0] = ax.scatter(frame_data[0, :], -frame_data[2, :], -frame_data[1, :], color='tab:blue')

        idx = 1
        for pse in pose_connections:
            plot[idx] = ax.plot3D([frame_data[0, pse[0]], frame_data[0, pse[1]]],
                                  [-frame_data[2, pse[0]], -frame_data[2, pse[1]]],
                                  [-frame_data[1, pse[0]], -frame_data[1, pse[1]]],
                                  color='k', lw=1)[0]
            idx += 1

        if rotate_animation:
            ax.view_init(elev=10., azim=120 + (360 / data.shape[-1]) * i)

        return figure,

    # Animate
    anim = animation.FuncAnimation(figure, animate, init_func=init,
                                   frames=144, interval=20, blit=True)

    plt.close()

    return anim
    
    
fig = plt.figure()
fig.set_size_inches(5, 5, True)
ax = fig.add_subplot(projection='3d')

anim = time_animate(data, fig, ax)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('im.mp4', writer=writer)

name = "wireframe.mp4"
writergif = animation.PillowWriter(fps=30)
anim.save('filename.gif',writer=writergif)



    
    


