import numpy as np
import json
import av
import string
import secrets
import os

from typing import (
    Dict, List, Union, Sequence, Mapping, Any, Optional
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_video(
    file_name: str, 
    frames: Union[List[np.ndarray], bytes], 
    width: int = 640, 
    height: int = 360, 
    fps: int = 20
) -> None:
    """Write video frames to video files. """
    with av.open(file_name, mode="w", format='mp4') as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

class Recorder:
    '''
    Recorder is used to record the trajectory of the agent. 
    The  
    '''
    def __init__(
        self, 
        root: dir,
        height: int = 360, 
        width: int = 640, 
        fps: int = 20,
        **kwargs, 
    ):
        self.root = root
        self.height = height
        self.width = width
        self.fps = fps 

    
    def save_trajectory(
        self, 
        video: Union[List[np.ndarray], bytes], 
        actions: List, 
        infos: List[Dict[str, Any]],
    ) -> None:
        '''
        Record trajectory with video, actions and infos. 
        Args:
            video: list of frames or bytes of video (encoded by pyav). 
            actions: list of actions. 
            infos: list of infos.
        Result:
            Generate a file name and save trajectories into `root/video`, 
            `root/actions`, and `root/infos` directories.
            For example, if the name is "abc1234_xx", then the video file 
            will be saved as `root/video/abc1234_xx.mp4`, the actions will
            be saved as `root/actions/abc1234_xx.json`. 
        '''

        # align video, actions and infos

        len_video, len_actions, len_infos = len(video), len(actions), len(infos)
        assert len_video == len_actions == len_infos, "The length of video, actions, and infos are different"

        # add index 
        actions_dict = dict()
        for i in range(len(actions)):
            actions_dict[i] = actions[i]
        infos_dict = dict()
        for i in range(len(infos)):
            infos_dict[i] = infos[i]
            del infos_dict[i]['pov']   

        # json file
        actions_json = json.dumps(actions_dict, cls=NumpyEncoder)
        infos_json =json.dumps(infos_dict, cls=NumpyEncoder)

        # save data
        file_name = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(10))
        if not os.path.exists(self.root + 'actions/'):
            os.makedirs(self.root + 'actions/', exist_ok=True)
        with open(self.root + 'actions/' + file_name + '.json', "w") as file:
            file.write(actions_json)

        if not os.path.exists(self.root + 'infos/'):
            os.makedirs(self.root + 'infos/', exist_ok=True)
        with open(self.root + 'infos/' + file_name + '.json', "w") as file:
            file.write(infos_json)

        if not os.path.exists(self.root + 'video/'):
            os.makedirs(self.root + 'video/', exist_ok=True)
        write_video(self.root + 'video/' +  file_name + '.mp4', video, self.width, self.height, self.fps)
