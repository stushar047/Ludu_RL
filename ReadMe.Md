Class Project: Reinforcment Learning: Learning to play Ludu using LuduPy (https://github.com/SimonLBSoerensen/LUDOpy) and Yaral Q-Learning for Ludu (https://github.com/NDurocher/YARAL). 



Ludu file has four main files:
1. Env.py
2. Agent.py
5. Hyperparamers.py
3. train_test.py
4. run.py

1. Env: The main idea of the Env file is to create the state and input those states to the environment. 
In order to do that, it has two main methods:

a. get_state(self, player_i, player_pieces, enemy_pieces, move_pieces): It takes the position of the player number, 
player_pieces, enemy_pieces, move_pieces and find either the piece is in home, goal_zone, safe or danger state.

 
b. get_state_raw(player_pieces, enemy_pieces): For raw, the state size is (8,60).
It just taken player pieces and enemy pieces (for a single enemy) and one hot encode the players position (0-59)
to the first 4 rows for four pieces and enemys position to the 5-8th row.

2. Agent.py: This file contains agent class, replay buffer class and network class.

a. Agent: The agent class has act method which takes a state and return the action using the local network and current movable pieces.

b. Replay Buffer: It is basically a memory to collect training data

c. Two types of network has been created - small network of 1 hidden layer and deep network of 4 hidden layers.

3. Hyperparameters: The hyperparameters can be modified from the hyperparameters.py file. Only the training_game 
and the testing_game can be given as a command terminal argument.

4. train_test.py:
First of all, this calls the gamming environment to create an episode. Then, the episodes data are modified to create the states.
After that, the agent acts based on the states and the local model which return the action. For reward, only negative -10 reward for overshotting,
+50 reward for killing opponent pices, -50 for opponent killing my players piece, +100 reward for winning the game and -100 reward for losing the 
game. Again, based on the dice and the action, the env finds out the next state and wether there is a winner (done). The tuple 
(current state,action,reward,next_state,done) is given the agent to take a step to store the experience in a replay buffer and 
learn after certain episodes.

5. run.py: This file runs the code with terminal argument training_game, testing_game, mode

In order to see the performance of the agent, run this file as the testing mode (mode=2)

Terminal commad: python run.py training_game testing_game mode. Here the model file also can be changed.

example: python run.py 10000 10000 2
 

   

 

