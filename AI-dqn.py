import numpy as np
import gym
import random
import time
import math
import copy
from Model import Net 
from PIL import Image
from Data_Memory import DataMemory
from skimage.color import rgb2gray
from skimage.transform import resize

h, w = 100, 100

def get_image(env):
    img = env.render(mode = 'rgb_array')
    #img = np.transpose(img, (2,0,1))
    return make_img(img)

def make_img(img, h=100, w=100):
    img = np.array(img, dtype=np.uint8)
    img = resize(rgb2gray(img), (h,w))
    return img
    '''
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
    '''

def image_debug(env, h=100, w=100):
    img = get_image(env)*255
    img = Image.fromarray(img)
    img.show()
                

envlist = []
envlist.append(gym.make('CartPole-v0'))
#envlist.append(gym.make('MountainCar-v0'))

model = Net()
model_freezed = copy.deepcopy(model)
print(model)
print(model_freezed)
done = False
tot_reward = 0
env = envlist[0]

eps_s = 1
eps_e = 0.1
play_time = 2000000
display_step = 200

gamma = 0.9
mb_size = 16
frame_skip = 2
input_num = 3
freeze_step = 5000

D = DataMemory(capacity = 1000)

Log = open('Log.txt', 'w')
Log.close()
Log = open('Log_train.txt', 'w')
Log.close()

def collect_data(env, gamma):
    env.reset()
    image_debug(env)
    img_old = get_image(env)
    state_old = []
    for i in range(input_num): state_old.append(img_old)
    print(np.array(state_old).shape)
    for t in range(play_time):
        if t % display_step == 0: run(env)
        if t % freeze_step == 0: model_freezed = copy.deepcopy(model)
        epsilon = eps_e+(eps_s-eps_e)*(t/play_time)
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
        D.push((state_old, action, reward, state_new, done))
        state_old = state_new
        train(env, mb_size = mb_size)
        if done:
            env.reset()
            img_old = get_image(env)
            state_old = []
            for i in range(input_num): state_old.append(img_old)
    return D

def train(env, mb_size = 32):
    Log = open("Log_train.txt", "a")
    targets = np.zeros((mb_size, env.action_space.n))
    inputs = np.zeros((mb_size, input_num, h, w))
    #print("Training minibatch ") 
    minibatch = D.sample(mb_size)
    print("Sampling data")
    for i in range(len(minibatch)):
        #print("Sampling data["+str(i)+"]")
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
        inputs[i] = state
        Q_sa = copy.deepcopy(model_freezed.predict(state_new))
        targets[i] = copy.deepcopy(model_freezed.predict(state))
        if done:
            targets[i][action] = reward
        else:
            targets[i][action] = reward + gamma * np.max(Q_sa)
    print(targets[0])
    loss = model.train(inputs, targets)
    Log.write("Trained minibatch: loss = "+str(loss.data[0])+"\n")   
    print("Trained minibatch: loss = "+str(loss.data[0]))     
    print("")

def run(env):
    Log = open("Log.txt", 'a')
    tot_reward = 0
    env.reset()
    img_old = get_image(env)
    state_old = []
    for i in range(input_num): state_old.append(img_old)
    for t in range(play_time):
        Log.write("Step "+str(t)+" started"+"\n")
        print("Step "+str(t)+" started")
        #time.sleep(0.01)
        Q = model.predict(state_old)
        print(Q)
        action = np.argmax(Q)
        Log.write(str(Q)+"\n")
        Log.write("Action : "+str(action)+"\n")
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
    Log.write("Total reward : "+str(tot_reward)+"\n\n")
    Log.close()
    return tot_reward
#print(run(env))
run(env)
for env in envlist:
    data = collect_data(env, gamma)
    print(run(env))
