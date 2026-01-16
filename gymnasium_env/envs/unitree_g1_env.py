import gymnasium as gym
import numpy as np
from mujoco import MjModel,MjData,viewer,mj_step
import time


class UnitreeG1Env(gym.Env):
    def __init__(self, render_mode=None):
        self.model = MjModel.from_xml_path(
            "./mujoco_menagerie/unitree_g1/scene_mjx.xml"
        )
        self.data = MjData(self.model)

        # Állapotok
        self.qpos = self.data.qpos
        self.qvel = self.data.qvel
        self.ctrl = self.data.ctrl
        self.dim_qpos = len(self.qpos)
        self.dim_qvel = len(self.qvel)
        self.dim_ctrl = self.model.nu

        self.ctrl_history=[0]
        self.initial_qpos = self.qpos.copy()
        self.initial_qvel = self.qvel.copy()
        self.target_qpos = self.qpos.copy()
        self.target_qvel = self.qvel.copy()



        self.actuator_ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.actuator_ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)

        self.action_space = gym.spaces.Box(-1,1,(self.dim_ctrl,),dtype=np.float32)   
        self.observation_space=gym.spaces.Box(-np.inf,np.inf, (self.dim_qpos + self.dim_qvel + self.dim_ctrl,) ,dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self.qpos, self.qvel,self.ctrl]).astype(np.float32)


    def _get_info(self):
        return {
            "distance_qpos": np.linalg.norm(
                self.qpos-self.target_qpos, ord=1
            ),
            "distance_qvel": np.linalg.norm(
                self.qvel-self.target_qvel, ord=1
            )
        }
    def _get_reward(self):
        reward_qpos = np.linalg.norm(self.qpos - self.target_qpos)
        reward_qvel = np.linalg.norm(self.qvel - self.target_qvel)
        reward_ctr = np.linalg.norm(self.ctrl)
        return -(reward_qpos+reward_ctr)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = self.target_qpos
        self.data.qvel[:] = self.target_qvel
        self.data.ctrl[:] = 0.0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    def step(self, action):
       
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.actuator_ctrl_low + (action + 1.0) * 0.5 * (self.actuator_ctrl_high - self.actuator_ctrl_low)
        self.ctrl[:] = ctrl
        self.ctrl_history.append(ctrl)  
        mj_step(self.model, self.data)

        # Observation, reward, info
        obs = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()
        base_z = self.data.qpos[2]
        terminated = base_z < 0.76
        return obs, reward, terminated, False, info



    def render(self):
        """
        Replay render: végigmegyünk a ctrl_history-n,
        minden lépésnél frissítjük a robotot a history-ból.
        """
        # Start time
        start = time.time()
        self.reset()  # Reseteljük a robotot
        self.viewer = viewer.launch_passive(self.model, self.data) 
        

        i = 0
        while  self.viewer.is_running():
            if (i>=len(self.ctrl_history)):
                i=len(self.ctrl_history)-1
            # Frissítjük a controlt
            self.ctrl[:] = self.ctrl_history[i]

            mj_step(self.model, self.data)

            self.viewer.sync()


            i += 1
        self.reset()
   



        

    def close(self):
      pass
