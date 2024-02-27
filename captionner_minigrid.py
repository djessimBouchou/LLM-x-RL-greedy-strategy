import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, STATE_TO_IDX, OBJECT_TO_IDX

from minigrid.envs import EmptyEnv
from minigrid.wrappers import FullyObsWrapper


class CaptionnerGT():
    def __init__(self):
        self.idx_to_object = IDX_TO_OBJECT
        self.object_to_idx = OBJECT_TO_IDX
        self.idx_to_state = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

    def caption(self, obs):
        """
        Observation : 'direction': Discrete(4), 'image': Box(0, 255, (_, _, 3), uint8), 'mission': MissionSpace
        where image[2] : OBJECT_IDX, COLOR_IDX, STATE)
        """
        image = obs["image"]

        res = ""
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if self.idx_to_object[image[i, j][0]] not in ["empty", "wall"]:
                    res += f"There is a {self.idx_to_object[image[i, j][0]]} at location ({i}, {j})\n"

        res += "All the other spaces are empty.\n\n"

        oriention_agent = obs["direction"]
        idx_to_direction = {0:"west",1:"south",2:"east",3:"north"}
        
        res += "The agent is currently facing the {}.\n\n".format(idx_to_direction[oriention_agent])
        return res


if __name__ == '__main__':
    captionner = CaptionnerGT()
    print(captionner.idx_to_object)
    print(captionner.idx_to_state)

    env = EmptyEnv(size = 20)
    env = FullyObsWrapper(env)

    obs, _ = env.reset()
    print("OK")