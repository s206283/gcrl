import gym
from robotics_rl_srl.environments.kuka_gym.kuka_button_gym_env import KukaButtonGymEnv
from robotics_rl_srl.environments.mobile_robot.mobile_robot_env import MobileRobotGymEnv
import numpy as np
from PIL import Image


class SRLEnv(gym.Env):
    keys = ['kuka', 'mobile']

    def __init__(self, action_repeat=1, environment='kuka',
                 srl_model="raw_pixels", height=84, width=84, renders=False,
                 is_discrete=False, force_down=False, render_kwargs=None):
        assert environment in self.keys
        self.environment = environment
        if self.environment == 'kuka':
            self.env = KukaButtonGymEnv(renders=renders, is_discrete=is_discrete, force_down=force_down,
                                            srl_model=srl_model)
        elif self.environment == 'mobile':
            self.env = MobileRobotGymEnv(renders=renders, is_discrete=is_discrete, srl_model=srl_model)
        #self.env = gym.make(env_id)
        self.action_repeat = action_repeat

        self.srl_model = srl_model

        self.render_kwargs = dict(
            width=width,
            height=height,
            depth=False,
            camera_name='track',
        )

        if render_kwargs is not None:
            self.render_kwargs.update(render_kwargs)

        if self.srl_model == "raw_pixels":
            obs_shape = (3, self.render_kwargs['height'], self.render_kwargs['width'])
            self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)
        else:
            self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space
        self._max_episode_steps = self.env.max_steps

    def _preprocess_obs(self, obs):
        if self.srl_model == "raw_pixels":
            #image = self.env.sim.render(**self.render_kwargs)[::-1, :, :]
            #print(type(obs[0]))
            #print(obs[0].shape)
            obs = np.array(obs).astype("uint8")
            #obs = np.array(obs[0]*255).astype("uint8")
            #print(obs)
            image = self.transform(obs)
            obs = np.transpose(image, [2, 0, 1])
        return obs

    def transform(self, obs):
        pixel = Image.fromarray(obs)
        resize = pixel.resize((self.render_kwargs['height'], self.render_kwargs['width']))
        #resize = np.array(resize).astype("float")
        resize = np.array(resize).astype("uint8")
        #feature = resize.transpose((2, 0, 1))

        #return feature
        return resize

    def step(self, action):
        sum_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, distance = self.env.step(action)
            sum_reward += reward
            if done:
                break
        return self._preprocess_obs(obs), sum_reward, done, distance

    def reset(self):
        obs = self.env.reset()
        return self._preprocess_obs(obs)

    def render(self, mode='rgb_array'):
        obs = self.env.render(mode=mode)
        obs = np.array(obs).astype("uint8")
        obs = Image.fromarray(obs)
        obs = obs.resize((self.render_kwargs['height'], self.render_kwargs['width']))
        #resize = np.array(resize).astype("float")
        obs = np.array(obs).astype("uint8")
        return obs

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.__del__()

    def get_goal_image(self):
        success_samples = []
        self.reset()
        while len(success_samples) < 10:
            self.reset()
            for i in range(self._max_episode_steps-1):
                #obs, reward, done, info = env.step(env.action_space.low)
                if self.environment == 'kuka':
                    action = self.env.action_space.sample()
                elif self.environment == 'mobile':
                    action = np.array([1, -0.1]) #mobile_navigation
                obs, reward, done, distance = self.step(action)
                if self.environment == 'kuka':
                    if done:
                        success_samples.append(obs)
                        print(obs.shape)
                        print(action)
                        break
                elif self.environment == 'mobile':
                    if reward:
                        success_samples.append(obs)
                        print(obs.shape)
                        #print(obs)
                        break

        return success_samples
