import numpy as np
import random 
from collections import namedtuple, deque 
import copy
import ludopy_v2 as ludopy

class Env:
    
    def __init__(self):
        self.tiles = [[1, 9, 22, 35, 48, 53], [5, 12, 18, 25, 31, 38, 44, 51], [53, 54, 55, 56, 57, 58], [59]]
        self.delta = 0.6
        self.LR=0.4
        self.R=np.array([0.25,0.0001,0.9,0.5,0.4,0.2,0.5,0,0.4])
        self.home = 0
        self.goal_zone = [53, 54, 55, 56, 57, 58, 59]
        self.safe = [1, 9, 22, 35, 48]
        self.danger = [14, 27, 40]
        self.epsilon = 0.1
    
    
    def get_state(self, player_i, player_pieces, enemy_pieces, move_pieces):
        
        num_pieces,action_index,action_value,state = len(move_pieces),np.ones(4)*-1,np.ones(4)*-1,np.arange(0,16,4)
        
        self.state_onehot=np.zeros(16);
        
        if num_pieces == 0:
            state = 0 + 4 * player_i
            self.state1 = np.arange(0,16,4)
        
        else:
            
            for i in range(0, num_pieces):
                       
                i = move_pieces[i]
                
                state[i] = self.findstate(i, player_pieces, enemy_pieces) ##states for this pieces    
            
            self.state1 = state
            
            for i in self.state1:

                self.state_onehot[i]=1;                                            

    
    def get_state_raw(self, player_pieces, enemy_pieces):
        
        state = np.zeros((8,60))
        
        adj_enemy = self.adjustenemy(enemy_pieces)
        
        idx=np.argmax(np.sum(adj_enemy,axis=1))
        
        enemy_spec=enemy_pieces[idx]; 
            
        for i in range(4):

            state[i,player_pieces[i]]=1 
            
            state[i+4,enemy_spec[i]]=1 

        self.state=state.reshape(-1,)
        
    def findstate(self, i, player_pieces, enemy_pieces):

        adj_enemy = self.adjustenemy(enemy_pieces)

        if player_pieces[i] == self.home:
            state = 0 + 4 * i
            
        elif self.ismember(self.goal_zone, player_pieces[i]) == 1:
            state = 1 + 4 * i
            
        elif self.ismember(self.safe, player_pieces[i]) == 1:
            state = 2 + 4 * i
            
        else:
            if self.iswithin(player_pieces[i],adj_enemy) or self.ismember(self.danger, player_pieces[i]):
                state = 3 + 4 * i
            else:
                state = 2 + 4 * i

        return state

    def iswithin(self, player, enemy_pieces):

        if self.ismember(enemy_pieces, player-1):
            return 1
        elif self.ismember(enemy_pieces, player-2):
            return 1
        elif self.ismember(enemy_pieces, player-3):
            return 1
        elif self.ismember(enemy_pieces, player-4):
            return 1
        elif self.ismember(enemy_pieces, player-5):
            return 1
        elif self.ismember(enemy_pieces, player-6):
            return 1
        else:
            return 0

    def adjustenemy(self, enemy):
        "Adjust all pices with respect to player 0"
        adj_enemy = np.zeros(enemy.shape)
        for i in range(enemy.shape[0]): ##Number of enemy
            for j in range(enemy.shape[1]): ## Piece of enemy
                if (enemy[i, j] != 0) & (enemy[i, j] < 54):
                    adj_enemy[i, j] = enemy[i, j] + (i + 1) * 13
                    if (adj_enemy[i, j] / 53) >= 1:
                        adj_enemy[i, j] += -52
        return adj_enemy

    def ismember(self, A, B):
        w = [np.sum(a == B) for a in A]
        return np.sum(w)
                     
                