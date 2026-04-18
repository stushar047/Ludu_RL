import numpy as np
import random 
from collections import namedtuple, deque 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    
    def __init__(self, state_size, action_size, seed,LR,BUFFER_SIZE,BATCH_SIZE,GAMMA,TAU,UPDATE_EVERY,loss,layers,net_type="deep"):
        
        self.BUFFER_SIZE = BUFFER_SIZE  
        self.BATCH_SIZE = BATCH_SIZE         
        self.GAMMA = GAMMA           
        self.TAU = TAU            
        self.UPDATE_EVERY = UPDATE_EVERY 
        self.LR=LR
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        if net_type=="deep":
            self.qnetwork_local = QNetwork_deep(state_size, action_size, seed, layers)
            self.qnetwork_target = QNetwork_deep(state_size, action_size, seed, layers)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed)
            self.qnetwork_target = QNetwork(state_size, action_size, seed)
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,BATCH_SIZE,seed)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done):
        self.memory.add(state, action, reward, next_step, done)
        self.t_step = (self.t_step+1)% self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory)>self.BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, self.GAMMA)
                
    def act(self, state, move_pieces, dice, eps = 0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        action_values1 = np.squeeze(action_values.cpu().data.numpy())
        
        move_pieces_=np.zeros(4)
        
        if len(move_pieces)>0:

                for m in move_pieces:

                    move_pieces_[m]=1

        self.qnetwork_local.train()
        
#         if ((dice == 6) & (sum(move_pieces_)<4)):
#             movable_piece=np.random.choice(np.where(move_pieces_==0)[0])
            
#         else:   

        if ((sum(action_values1[1:]*move_pieces_)==0) or (action_values1[0]==1)):
            movable_piece=-1    
        else:
            aa=action_values1[1:]*move_pieces_

            idx= np.where(aa!=0)[0]

            val=np.argmax(aa[idx])

            movable_piece=idx[val]
            
        if random.random() > eps:
            return movable_piece,np.argmax(action_values1)
        else:
            return movable_piece,random.choice(np.arange(self.action_size))
           
            
    def learn(self, experiences, gamma,loss="MSE"):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        
        if loss=="MSE":

            criterion = torch.nn.MSELoss()
            
        elif loss=="L1Loss": 
            
            criterion = torch.nn.L1Loss()
            
        elif loss=="SmoothL1Loss": 
            
            criterion = torch.nn.SmoothL1Loss()
            
        elif loss=="LNLLLoss": 
            
            criterion = torch.nn.NLLLoss()
        
        elif loss=="CrossEntropyLoss": 
            
            criterion = torch.nn.CrossEntropyLoss()

        self.qnetwork_local.train()
        
        self.qnetwork_target.eval()

        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (gamma* labels_next*(1-dones))
        
        loss = criterion(predicted_targets,labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        self.optimizer.step()
        self.scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.TAU)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, seed, fc1_unit=64,
                 fc2_unit = 64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    
class QNetwork_deep(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, seed, layers=[64,64,64,64,64,64,64,64,64,64]):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super(QNetwork_deep,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],layers[2])
        self.fc4 = nn.Linear(layers[2],layers[3])
#         self.fc5 = nn.Linear(layers[3],layers[4])
#         self.fc6 = nn.Linear(layers[4],layers[5])
#         self.fc7 = nn.Linear(layers[5],layers[6])
#         self.fc8 = nn.Linear(layers[7],layers[8])
#         self.fc9 = nn.Linear(layers[8],layers[9])
        self.fc5 = nn.Linear(layers[3],action_size)
        
    def forward(self,x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = F.relu(self.fc4(x3))
#         x5 = F.relu(self.fc5(x4))
#         x6 = F.relu(self.fc6(x5))
#         x7 = F.relu(self.fc7(x6))
#         x8 = F.relu(self.fc8(x7))
#         x9 = F.relu(self.fc9(x8))
        return self.fc5(x4)    