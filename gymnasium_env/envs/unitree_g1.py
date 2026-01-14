from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from mujoco import MjModel,MjData
from mujoco import viewer



class UnitreeG1Env(gym.Env):
 
    def __init__(self, render_mode=None):
        # Mujoco model betöltése
        self.model = MjModel.from_xml_path(
            "/home/tomi/AIMotionLab-Virtual/scripts/unitree_g1/mujoco_menagerie/unitree_g1/scene_mjx.xml"
        )
        self.data = MjData(self.model)

        # Állapotok
        self.qpos = self.data.qpos
        self.qvel = self.data.qvel
        self.ctrl = self.data.ctrl
        self.initial_qpos = self.qpos.copy()
        self.initial_qvel = self.qvel.copy()

       
        self.action_space = gym.spaces.Box(-0.4,0.4, (self.model.nu,) ,dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf, (len(self.qpos)*len(self.qvel),) ,dtype=np.float32)

     

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
      
        pass

    def step(self, action):
     

        return observation, reward, terminated, False, info

    def render(self):
        # Csak human mód
        if not hasattr(self, "_viewer"):
            # Egyszeri ablaknyitás
            self._viewer = viewer.launch(self.model, self.data)
        else:
            # Frissítjük a jelenlegi frame-et
            self._viewer.render()

    def _render_frame(self):
      

      pass

    def close(self):
      pass