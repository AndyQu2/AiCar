import cv2
import gym
import numpy as np
import torch
import gym_donkeycar

from AutoDrive.model import AutoDriveNet

env = gym.make('donkey-generated-roads-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoDriveNet()
model.load_state_dict(torch.load('output\\model.pth'))
model.to(device)

env.reset()
action = np.array([0, 0.1])

img, reward, done, info = env.step(action)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

model.eval()
while True:
    img = torch.from_numpy(img.copy()).float()
    img /= 255.0
    img = img.permute(2, 0, 1)
    img.unsqueeze_(0)
    img.to(device)

    factor = -1
    with torch.no_grad():
        steering_angle = (model(img).squeeze(0).cpu().detach().numpy())[0]
        steering_angle = steering_angle * factor
        if steering_angle * factor < -1:
            steering_angle = -1
        elif steering_angle * factor > 1:
            steering_angle = 1
        else:
            steering_angle = steering_angle * factor

    action = np.array([steering_angle, 0.1])
    img, reward, done, info = env.step(action)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)