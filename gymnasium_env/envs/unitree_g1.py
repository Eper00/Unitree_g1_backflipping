import gymnasium as gym
import numpy as np
from mujoco import MjModel,MjData,viewer,mj_step
import time
class UnitreeG1Env(gym.Env):
    def __init__(self, render_mode=None):
        self.model = MjModel.from_xml_path(
            "/home/tomi/AIMotionLab-Virtual/scripts/unitree_g1/mujoco_menagerie/unitree_g1/scene_mjx.xml"
        )
        self.data = MjData(self.model)

        # Állapotok
        self.qpos = self.data.qpos
        self.qvel = self.data.qvel
        self.ctrl = self.data.ctrl
        self.dim_qpos = len(self.qpos)
        self.dim_qvel = len(self.qpos)
        self.dim_ctrl = self.model.nu
        self.initial_qpos = self.qpos.copy()
        self.initial_qvel = self.qvel.copy()
        self.target_qpos = self.qpos.copy()
        self.target_qvel = self.qvel.copy()

        actuator_joints = self.model.actuator_trnid[:, 0]

        low = []
        high = []
        self.ctrl_history=[]
        self.ctrl_history.append(self.data.ctrl)
        for j in actuator_joints:
            low.append(self.model.jnt_range[j, 0])
            high.append(self.model.jnt_range[j, 1])

        low = np.array(low, dtype=np.float32)
        high = np.array(high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )   
        self.observation_space=gym.spaces.Box(-np.inf,np.inf, (len(self.qpos) + len(self.qvel) + len(self.ctrl),) ,dtype=np.float32)

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
        return -(reward_qpos+0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.data.qpos[:] = self.target_qpos
        self.data.qvel[:] = self.target_qvel
        self.data.ctrl[:] = 0.0

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # Apply action
        self.data.ctrl[:] = action
        self.ctrl_history.append(self.data.ctrl.copy())  # copy kell, külön tároljuk minden step-et

        # MuJoCo léptetés
        mj_step(self.model, self.data)

        # Observation, reward, info
        obs = self._get_obs()
        reward = self._get_reward()
        info = self._get_info()

        base_z = self.data.qpos[2]
        terminated = base_z < 0.25

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

            # MuJoCo léptetés
            mj_step(self.model, self.data)

            # Viewer frissítés
            self.viewer.sync()

            # Lassítás a vizualizációhoz

            i += 1
        self.reset()
   



        

    def close(self):
      pass
