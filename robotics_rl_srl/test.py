import gym
import pybulletgym.envs
from environments.kuka_gym.kuka_button_gym_env import KukaButtonGymEnv
from environments.mobile_robot.mobile_robot_env import MobileRobotGymEnv
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 環境の生成
#env = gym.make('KukaButtonGymEnv')
env = KukaButtonGymEnv(renders=True)
#env = MobileRobotGymEnv(renders=True)
env.render(mode="rgb_array")
env.reset()

resize = T.Compose([T.ToPILImage(),
                    #T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

# 空間の出力
def print_spaces(label, space):
   # 空間の出力
   print(label, space)

   # Box/Discreteの場合は最大値と最小値も表示
   if isinstance(space, Box):
       print('    最小値: ', space.low)
       print('    最大値: ', space.high)
   if isinstance(space, Discrete):
       print('    最小値: ', 0)
       print('    最大値: ', space.n-1)

#print_spaces('状態空間: ', env.observation_space)
#print_spaces('行動空間: ', env.action_space)
#print(env.action_space.high)
#print(env.action_space.low)
"""
env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()
"""


done = False

# ランダム行動
while True:
    #env.render(mode="human")
    #acton = np.array([-0.4]*env.action_space.shape[0])
    action = env.action_space.sample()
    #action = np.array([10, 0, 0, 0, 0, 0, 0])
    #action = np.array([1, -0.1]) #mobile_navigation
    observation, reward, done, info = env.step(action)
    #observation, reward, done, info = env.step(acton)
    print(action)
    print(observation)
    #print(observation.shape)
    #print(reward)
    print(info)
    print(done)

    if reward:
        plt.figure()
        plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                   interpolation='none')
        plt.title('goal image')
        plt.show()


    if done:
        plt.figure()
        plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
                   interpolation='none')
        plt.title('goal image')
        plt.show()
