import numpy as np
import gym
import random
import time
import math
from Model import Net 
from PIL import Image

h, w = 100, 100

def get_image(env):
    img = env.render(mode = 'rgb_array')
    img /= 255
    img = np.transpose(img, (2,0,1))
    return make_img(img)

def make_img(img, h=100, w=100):
    n = img.shape[0]
    h0 = img.shape[1]
    w0 = img.shape[2]
    newimg = np.zeros((h,w))
    for k in range(n):
        for i in range(h):
            ii = int(h0*i/h)
            for j in range(w):
                jj = int(w0*j/w)
                newimg[i][j]=float(img[k][ii][jj])/n
    return newimg

def image_debug(env, h=100, w=100):
    img = env.render(mode = 'rgb_array')
    print(img.shape)
    n = 3
    newimg=np.zeros((h,w,n), dtype=np.uint8)
    h0 = img.shape[0]
    w0 = img.shape[1]
    for k in range(n):
        for i in range(h):
            ii = int(h0*i/h)
            if i!=ii: print(i,ii)
            for j in range(w):
                jj = int(w0*j/w)
                if j!=jj: print(j,jj)
                newimg[i][j][k]=img[ii][jj][k]
    img = Image.fromarray(newimg, 'RGB')
    img.show()
                

envlist = []
envlist.append(gym.make('CartPole-v0'))
#envlist.append(gym.make('MountainCar-v0'))

model = Net()
print(model)
done = False
tot_reward = 0
env = envlist[0]

eps_s = 0.9
eps_e = 0.05
repeat_num = 50
play_time = 200
tot_count = 0.
tot_play = repeat_num * play_time

gamma = 0.9
mb_size = 5
mb_num = 30
frame_skip = 3
input_num = 3

tot_count

def collect_data(env, gamma):
    env.reset()
    img_old = get_image(env)
    state_old = []
    for i in range(input_num): state_old.append(img_old)
    print(np.array(state_old).shape)
    D = []
    for t in range(play_time):
        tot_count += 1
        epsilon = eps_e+(eps_s-eps_e)*math.exp(-tot_count/tot_play)
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
            obs, reward, done, info = env.step(action)
        img_new = get_image(env)
        state_new = []
        for i in range(input_num-1):
            state_new.append(state_old[i+1])
        state_new.append(img_new)
        #if done: break
        D.append((state_old, action, reward, state_new, done))
        state_old = state_new
        if done:
            env.reset()
            img_old = get_image(env)
            state_old = []
            for i in range(input_num): state_old.append(img_old)
    return D

def train(D, env, mb_num, mb_size):
    targets = np.zeros((mb_size, env.action_space.n))
    inputs = np.zeros((mb_size, input_num, h, w))
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
            
            inputs[i] = state
            Q_sa = model.predict(state_new)
            targets[i] = model.predict(state)
            #print(inputs)
            print(targets)
            if done:
                targets[i] = reward
            else:
                targets[i] = reward + gamma * np.max(Q_sa)
        loss = model.train(inputs, targets)    
        print("Trained minibatch "+str(step)+": loss = "+str(loss.data[0]))     
    print(reward)

def run(env):
    tot_reward = 0
    env.reset()
    img_old = get_image(env)
    state_old = []
    for i in range(input_num): state_old.append(img_old)
    for t in range(play_time):
        print("Step "+str(t)+" started")
        #time.sleep(0.01)
        Q = model.predict(state_old)
        print(Q)
        action = np.argmax(Q)
        print("AI action : " + str(action))
        print("")
        #for i in range(repeat_step):
        obs, reward, done, info = env.step(action)
        img_new = get_image(env)
        state_new = []
        for i in range(input_num-1):
            state_new.append(state_old[i+1])
        state_new.append(img_new)
        #if done: break
        state_old = state_new
        if done: break
        tot_reward += reward
    return tot_reward
#print(run(env))


for env in envlist:
    for step in range(repeat_num)
        data = collect_data(env, gamma)
        train(data, env, mb_num, mb_size)
        print(run(env))
