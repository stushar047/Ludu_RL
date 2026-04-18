import Hyperparameters as hyp
from Agent_ import *
from Env import *
from train_test import *
import copy
import sys
import time
import numpy as np
import argparse


#parser    
parser = argparse.ArgumentParser(description='Run the code')
parser.add_argument('training_game', type=int)
parser.add_argument('testing_game', type=int)
parser.add_argument('mode', type=int)
args = parser.parse_args()

BUFFER_SIZE = hyp.Val['BUFFER_SIZE']  #replay buffer size
BATCH_SIZE = hyp.Val['BATCH_SIZE']       # minibatch size
GAMMA = hyp.Val['GAMMA']             # discount factor
TAU = hyp.Val['TAU']               # for soft update of target parameters
LR = hyp.Val['LR']
net_type = hyp.Val['network']
UPDATE_EVERY = hyp.Val['UPDATE_EVERY']
layers=hyp.Val['layers']
loss=hyp.Val["loss"]
training_game=args.training_game
testing_game=args.testing_game
action_size=5

if state_=="Raw":
    state_size=480
    
elif state_=="Processed":
    state_size=16
    
else:
    EOFError("State Invalid")

agent=Agent(state_size=state_size,action_size=action_size,seed=0,LR=LR,BUFFER_SIZE=BUFFER_SIZE,BATCH_SIZE=BATCH_SIZE,\
              GAMMA=GAMMA,TAU=TAU,UPDATE_EVERY=UPDATE_EVERY,layers=layers,loss=loss,net_type=net_type)

if args.mode==1:

    print("Start Training")
    training(agent,training_game,load_model=True,filename="exp")

print("Start Testing")
testing(agent, testing_game, model_file="file_main.pth",mode="test")