import av
import cv2
import pickle
from jarvis.stark_tech.env_interface import MinecraftWrapper
from typing import Union
import numpy as np
from copy import deepcopy
import torch
    
    
def video2np(video_path):
    """format: THWC, RGB"""
    if isinstance(video_path, np.ndarray):
        return video_path
    if isinstance(video_path, torch.Tensor):
        return video_path.cpu().numpy()
    rgb_frame_list = []
    video_read_capture = cv2.VideoCapture(video_path)
    while video_read_capture.isOpened():
        result, frame = video_read_capture.read()
        if not result:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_list.append(rgb_frame)
    video_read_capture.release()
    video_nparray = np.array(rgb_frame_list, dtype=np.uint8)
    return video_nparray


def resize_varray(varray, h, w) -> np.ndarray:
    """THWC, resize to shape (h, w)"""
    assert len(varray.shape) == 4, str(varray.shape)
    T, H, W, C = varray.shape
    if isinstance(varray, torch.Tensor):
        varray = varray.cpu().numpy()
    reshaped = np.empty(shape=(T, h, w, C), dtype=varray.dtype)
    for t in range(T):
        reshaped[t] = cv2.resize(varray[t], dsize=(w, h))
    return reshaped
    

def np2video(array, width, height, video_out_path):
    if isinstance(array, np.ndarray):
        varray = array.astype(np.uint8)
    elif isinstance(array, torch.Tensor):
        varray = array.to('cpu').numpy().astype(np.uint8)
    container = av.open(video_out_path, mode='w', format='mp4')
    stream = container.add_stream('h264', rate=20)
    stream.width = width
    stream.height = height
    stream.pix_fmt = 'yuv420p'
    for frame in varray:
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def np2picture(array, picture_out_path):
    if isinstance(array, np.ndarray):
        varray = array.astype(np.uint8)
    elif isinstance(array, torch.Tensor):
        varray = array.to('cpu').numpy().astype(np.uint8)
    varray = cv2.cvtColor(varray, cv2.COLOR_RGB2BGR)
    cv2.imwrite(picture_out_path, varray)
    
def np2image(array, image_path):
    np2picture(array, image_path)


def add_actions_to_video(video: Union[str, np.ndarray], actions: Union[str, list, dict], dest_path: str, fps: int):
    
    if isinstance(actions, str):
        with open(actions, 'rb') as f:
            actions = pickle.load(f)
    if isinstance(video, np.ndarray):
        frames = deepcopy(video)
    else:
        if video.endswith('.mp4'):
            cap = cv2.VideoCapture(video)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            cap.release()
        elif video.endswith('.pkl'):
            with open(video, 'rb') as f:
                frames = pickle.load(f)

    container = av.open(dest_path, mode='w', format='mp4')
    stream = container.add_stream('h264', rate=fps)
    stream.width = 640
    stream.height = 360
    stream.pix_fmt = 'yuv420p'

    for frame_id in range(len(frames)):
        
        frame = frames[frame_id]
        if isinstance(actions, dict):
            keys = list(actions.keys())
            action = {k: actions[k][frame_id] for k in keys}
        elif isinstance(actions, list):
            if 'attack' in actions[0]:
                action = actions[frame_id]
            else:
                action = actions[frame_id]
                action = MinecraftWrapper.agent_action_to_env((action['buttons'], action['camera']))

        for row, (k, v) in enumerate(action.items()):
            color = (234, 53, 70) if (v != 0).any() else (249, 200, 14) 
            if k == 'camera':
                v = "[{:.2f}, {:.2f}]".format(v[0], v[1])
            cv2.putText(frame, f"{k}: {v}", (10, 25 + row*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    stream.close()
    container.close()
    
if __name__ == '__main__':
    ...