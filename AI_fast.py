import numpy as np
import gym
import random
import time
from Fast_Model import Fast_Net 
from PIL import Image

h, w = 100, 100

envlist = []
#envlist.append(gym.make('CartPole-v0'))
envlist.append(gym.make('MountainCar-v0'))

model = Fast_Net()
print(model)
done = False
tot_reward = 0
repeat_step = 5
play_time = 50000
env = envlist[0]

epsilon = 0.3
gamma = 0.9
mb_size = 50
mb_num = 1000
frame_skip = 3
input_num = 2

def collect_data(env, epsilon, gamma):
    obs = env.reset()
    print(obs)
    state_old = []
    for i in range(input_num): state_old.append(obs)
    print(np.array(state_old).shape)
    D = []
    for t in range(play_time):
        env.render()
        #image_debug(env)
        print("Step "+str(t)+" started")
        #time.sleep(0.01)
        if random.random() <= epsilon:
            action = random.randrange(0, env.action_space.n) 
            print("Random action : " + str(action))
        else:
            Q = model.predict(state_old)
            print(Q)
            action = np.argmax(Q)
            print("AI action : " + str(action))
        print("")
        #for i in range(repeat_step):
        for i in range(frame_skip):
            obs_new, reward, done, info = env.step(action)
        state_new = []
        for i in range(input_num-1):
            state_new.append(state_old[i+1])
        state_new.append(obs_new)
        #if done: break
        D.append((state_old, action, reward, state_new, done))
        state_old = state_new
        if done:
            obs = env.reset()
            state_old = []
            for i in range(input_num): state_old.append(obs)
    return D

def train(D, env, mb_num, mb_size):
    targets = np.zeros((mb_size, env.action_space.n))
    inputs = np.zeros((mb_size, 2, 2))
    for step in range(mb_num):
        print("Training minibatch "+str(step)) 
        minibatch = random.sample(D, mb_size)
        for i in range(0, mb_size):
            print("Sampling data["+str(step)+":"+str(i)+"]")
            state = minibatch[i][0]
            action = minibatch[i][1]
            reward = minibatch[i][2]
            state_new = minibatch[i][3]
            done = minibatch[i][4]
            #print(state) 
            inputs[i] = state
            Q_sa = model.predict(state_new)
            targets[i] = model.predict(state)
            print(inputs)
            #print(targets)
            if done:
                targets[i] = reward
            else:
                targets[i] = reward + gamma * np.max(Q_sa)
        loss = model.train(inputs, targets)    
        print("Trained minibatch "+str(step)+": loss = "+str(loss.data[0]))     
    print(reward)

def run(env):
    tot_reward = 0
    obs = env.reset()
    state_old = []
    for i in range(input_num): state_old.append(obs)
    for t in range(play_time):
        env.render()
        print("Step "+str(t)+" started")
        #time.sleep(0.01)
        Q = model.predict(state_old)
        print(Q)
        action = np.argmax(Q)
        print("AI action : " + str(action))
        print("")
        for i in range(frame_skip):
            obs, reward, done, info = env.step(action)
        state_new = []
        for i in range(input_num-1):
            state_new.append(state_old[i+1])
        state_new.append(obs)
        #if done: break
        state_old = state_new
        if done: break
        tot_reward += reward
    return tot_reward
#print(run(env))


for env in envlist:
    print(env.action_space.n)
    for step in range(5):
        data = collect_data(env, epsilon, gamma)
        train(data, env, mb_num, mb_size)
        print(run(env))
