from Agent_ import *
import Hyperparameters as hyp
from Env import *
import copy
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

state_=hyp.Val['State']
action_size=5

def training(agent, training_game=600,filename="exp",load_model=False,model_file="file_main.pth"):
    
    if load_model==True:
    
        agent.qnetwork_local.load_state_dict(torch.load(model_file))
        
        print("model loaded")
    
    tiles = [[1, 9, 22, 35, 48, 53], [5, 12, 18, 25, 31, 38, 44, 51], [53, 54, 55, 56, 57, 58], [59]]
    home = 0
    goal_zone = [53, 54, 55, 56, 57, 58, 59]
    safe = [1, 9, 22, 35, 48]
    danger = [14, 27, 40]
    epsilon = 0.1
    
    count=0
    
    scores = [] # list containing score from each episode
    episode=[];
    scores_window = deque(maxlen=100) # last 100 scores
    
    "Create an player"
    player0 = Env()
    
    "Player not participated in the game"
    ghosts = [2,3]
    
    "Start Counting time"
    start_time = time.time()
    
    player_wins=0
    
    enemy_wins = 0
    
    player_wins_=0
    
    player_wins_test=[]
    
    enemy_wins_= 0
    
    player_wins_100=[]
    
    score_100=[]
    
    "Iterating all games"
    for i in range(0, training_game): 
        
        ra=0
        
        g = ludopy.Game(ghost_players=ghosts)
        
        actions,overshoots,score,there_is_a_winner=0,0,0,False
        
#         print(f"Episode: {i}")
        
        while not there_is_a_winner:
            "Get a observation"
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, 
             there_is_a_winner), player_i = g.get_observation()

            if (player_i > 0):
                "For other players take random action unless overshoot"
                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                    if player_pieces[piece_to_move]+ dice > 59:
                        overshoots += 1
                else:
                    piece_to_move = -1
                "Check anyone wins"    
                _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
                
                if piece_to_move>=0:
                    
                    if enemy_pieces[0][piece_to_move]+ dice ==player_pieces[mp]:
                            ra = -100
                
                if there_is_a_winner == True:
                    ra = -100
                    enemy_wins = enemy_wins + 1
                    #print(ra)
                    
#                 print(f"Reward: {ra}")    
            else:
                
                if state_=='Raw':

                    player0.get_state_raw(player_pieces, enemy_pieces)

                    state=player0.state
                    
                else:
                    
                    player0.get_state(player_i, player_pieces, enemy_pieces, move_pieces)

                    state=player0.state_onehot
                    
                mp,action=agent.act(state,move_pieces,dice)

                _, new_move_pieces, new_P0, new_enemy, _, there_is_a_winner = g.answer_observation(mp)
                
                if mp>=0:
                    
                    if enemy_pieces[0][mp] ==player_pieces[mp]+dice: #Kill
                            ra = 100
                
                    if player_pieces[mp]+ dice > 59: ##Overshoot
                        ra = -10  
                        
                    if (tiles[3] == (player_pieces[mp] + dice)) or ((player_pieces[mp] + dice) in goal_zone): ##Goal or Goal Zone
                        ra = 0 
                        
                    if (player_pieces[mp] + dice in tiles[1]): ##Star
                        ra = 10
                        
                    if (player_pieces[mp] + dice in tiles[0]): ##Globe
                        ra = 5
                        
                    if (player_pieces[mp] + dice in player_pieces): ##protect
                        ra = 10     
                        
                    if (new_P0[mp]==player_pieces[mp]) & (len(move_pieces)>0): ##If move is made and no move is possible for that state
                        ra = -1
                    
                    if ((dice==6) & len(move_pieces)==len(new_move_pieces)): ##Open
                        ra = -1
                
                if there_is_a_winner == True:
                    ra = 100
                    player_wins = player_wins + 1
                
                if state_=='Raw':

                    player0.get_state_raw(player_pieces, enemy_pieces)

                    next_step=player0.state
                    
                else:
                    
                    player0.get_state(player_i, player_pieces, enemy_pieces, move_pieces)

                    next_step=player0.state_onehot 
                        
                done=there_is_a_winner

                agent.step(state, action, ra, next_step, done)
                
#                 print(f"Reward: {ra}")
                     
            score+= ra

        scores_window.append(score) ## save the most recent score
        
        scores.append(score) ## sae the most recent score
        
#         print(scores)
        
        episode.append(i) ## sae the most recent score
         
        if i %100==0:
            
            score_100.append(np.mean(scores_window))
            
            print("Online:")
            
            print('\rEpisode {}\tAverage Score in the last 100 episodes{:.2f}'.format(i,np.mean(scores_window)))
            
            print('Player Total Wins',player_wins,'Enemy Total wins', enemy_wins)
            
            player_wins_100.append(player_wins-player_wins_)
            
            print('Last 100 episodes Player Wins ',(player_wins-player_wins_),'Enemy wins', (enemy_wins-enemy_wins_))
            
            player_wins_=player_wins
    
            enemy_wins_= enemy_wins
            
            print("Offline:")
        
            player_wins_test.append(testing(agent, testing_game=100))
   
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(0,training_game,100),player_wins_100,'-b^')
    plt.plot(np.arange(0,training_game,100),player_wins_test,'-gv')
    plt.xlabel('The number of Episodes',fontsize=20)
    plt.ylabel('Number of win',fontsize=20)
    plt.legend(['Online','Offline'],fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(filename+'_win.jpg',dpi=600)
    
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(0,training_game,100),score_100,'-b^')
    plt.xlabel('The number of Episodes',fontsize=20)
    plt.ylabel('Average_Reward',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(filename+'_reward.jpg',dpi=600)
#     plt.show()
    
    df=pd.DataFrame({"Episodes":episode,"Scores":scores})
    
    df.to_csv(filename+'.csv')
    
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')       
   
    end_time = time.time()
    
    print("Total Time Taken:", int(end_time - start_time), "Seconds")
    
    print('Player Wins',player_wins,'Enemy wins', enemy_wins)
    
    return scores


def testing(agent, testing_game=100,model_file="checkpoint.pth",mode="train"):
    
    if mode=="test":
    
        agent.qnetwork_local.load_state_dict(torch.load(model_file))
    
    "Create an player"
    player0 = Env()
    
    "Player not participated in the game"
    ghosts = [2,3]
    
    "Start Counting time"
    start_time = time.time()
    
    player_wins=0
    
    enemy_wins = 0
    
    "Iterating all games"
    for i in range(0, testing_game): 
        
        g = ludopy.Game(ghost_players=ghosts)
        actions,overshoots,score,there_is_a_winner=0,0,0,False
        
        while not there_is_a_winner:
            "Get a observation"
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, 
             there_is_a_winner), player_i = g.get_observation()

            if (player_i > 0):
                "For other players take random action unless overshoot"
                if len(move_pieces):
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                    if player_pieces[piece_to_move]+ dice > 59:
                        overshoots += 1
                else:
                    piece_to_move = -1
                "Check anyone wins"    
                _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
                
                if there_is_a_winner == True:
                    enemy_wins = enemy_wins + 1

            else:
                
                if state_=='Raw':

                    player0.get_state_raw(player_pieces, enemy_pieces)

                    state=player0.state
                    
                else:
                    
                    player0.get_state(player_i, player_pieces, enemy_pieces, move_pieces)

                    state=player0.state_onehot   

                mp,action=agent.act(state,move_pieces,dice)

                _, _, new_P0, new_enemy, _, there_is_a_winner = g.answer_observation(mp)    
                
                if there_is_a_winner == True:
                    
                    player_wins = player_wins + 1
    
    end_time = time.time()
    
    print("Total Time Taken:", int(end_time - start_time), "Seconds")
    
    print('Player Wins',player_wins,'Enemy wins', enemy_wins)
    
    if mode=="train":
    
        return player_wins