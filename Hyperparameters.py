Val={
"State":"Processed",#Raw, else
"BUFFER_SIZE" : int(1e3),  #replay buffer size
"BATCH_SIZE": 32,      # minibatch size
"GAMMA": 0.7,            # discount factor
"TAU": 1e-3,              # for soft update of target parameters
"LR": 0.0001,              # learning rate
"UPDATE_EVERY": 20,
"network":"small", #deep,small
"layers":[64,64,64,64], #10 layers of deep network   
"loss":"SmoothL1Loss"} #L1,MSE,SmoothL1Loss,NLLLoss,CrossEntropyLoss