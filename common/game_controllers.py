import common.game_constants as game_constants
import common.game_state as game_state
import pygame

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms

class DQN(nn.Module):
  def __init__(self,input_size, output_size):
    super().__init__()
    self.encoder = torch.nn.Sequential(
        torch.nn.Linear(input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,64),
        torch.nn.ReLU(),
        torch.nn.Linear(64,32),
        torch.nn.ReLU(),
        torch.nn.Linear(32,output_size),
        torch.nn.Softmax(dim=1)
    )

  def forward(self,state:game_state.GameState):
    _,x=convert_input(state)
    x=self.encoder(x)
    return x


def convert_input(state:game_state.GameState):
  player=[state.PlayerEntity.entity.x, state.PlayerEntity.entity.y, state.PlayerEntity.entity.height, state.PlayerEntity.entity.width, 
          state.PlayerEntity.velocity.x, state.PlayerEntity.velocity.y, state.PlayerEntity.friction, state.PlayerEntity.acc_factor]
  goal=[state.GoalLocation.x, state.GoalLocation.y, state.GoalLocation.height, state.GoalLocation.width]
  enemies=[]
  for i in state.EnemyCollection:
    new_en=[i.entity.x, i.entity.y, i.entity.height, i.entity.width , i.velocity.x, i.velocity.y]
    enemies=enemies+ new_en
  bound= [state.Boundary.x, state.Boundary.y, state.Boundary.height, state.Boundary.width]
  curr= [state.Current_Observation.value]

  in_len=len(player+goal+enemies+bound+curr)

  return in_len,torch.tensor(player+goal+enemies+bound+curr).unsqueeze(0)



class KeyboardController:
    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        keys = pygame.key.get_pressed()
        action = game_state.GameActions.No_action
        if keys[pygame.K_LEFT]:
            action = game_state.GameActions.Left
        if keys[pygame.K_RIGHT]:
            action = game_state.GameActions.Right
        if keys[pygame.K_UP]:
            action = game_state.GameActions.Up
        if keys[pygame.K_DOWN]:
            action = game_state.GameActions.Down
    
        return action


class AIController:
### ------- You can make changes to this file from below this line --------------
    def __init__(self) -> None:
        self.model=DQN(self.get_len(),5)
        self.opt=optim.Adam(self.model.parameters(),lr=0.00001)
        self.epsilon=0.3
        # pass

    def get_len(self):
        state = game_state.GameState()
        in_len,_= convert_input(state)
        return in_len

    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        map={0:game_state.GameActions.No_action,1:game_state.GameActions.Up, 2:game_state.GameActions.Down, 3:game_state.GameActions.Left, 4:game_state.GameActions.Right}
        out= self.model(state)
        preds = torch.argmax(out, dim=1).numpy()[0]

        if np.random.uniform() < self.epsilon:
            # Random action
            action = np.random.choice(list(game_state.GameActions))
        else:
            action=map[preds]

        return action
      
    def TrainModel(self):

        self.discount_factor = 0.99
        epochs = 1000 # You might want to change the number of epochs
        map2={game_state.GameActions.No_action : 0, game_state.GameActions.Up :1 , game_state.GameActions.Down : 2, game_state.GameActions.Left :3, game_state.GameActions.Right :4}
        for _ in range(epochs):

            
            state = game_state.GameState()

            done = False
            while not done:
                self.opt.zero_grad()
                # Choose action with epsilon-greedy policy
                out=self.model(state).squeeze(0)
                # print(out.shape)
                action = self.GetAction(state)
                # print(action)
                act=map2[action]

                q=out[act].detach().numpy()

                # Take action and observe next state and reward
                obs = state.Update(action)

                out2=self.model(state).squeeze(0)
                action2 = self.GetAction(state)
                act2=map2[action2]

                q_dash=out2[act2].detach().numpy()

                player=[state.PlayerEntity.entity.x, state.PlayerEntity.entity.y]
                goal=[state.GoalLocation.x, state.GoalLocation.y]

                dist=0
                for i in range(2):
                    dist = dist+ (player[i]-goal[i])**2
                # print(dist)
                reward = 0
                if obs == game_state.GameObservation.Enemy_Attacked:
                    reward = -0.1
                    done = True
                elif obs == game_state.GameObservation.Reached_Goal:
                    reward = 1
                    done = True
                elif obs == game_state.GameObservation.Nothing:
                    if dist<1000:
                        reward=1000/dist
                    else:
                        reward = -(dist/100000)

                td_error = (reward + self.discount_factor * q_dash - q)**2
                td_error=torch.tensor(td_error).requires_grad_()
                # print("#########",td_error)

                td_error.backward()

                self.opt.step()

        # weights = self.model.state_dict()

        # # save the weights to a file
        # torch.save(weights, 'model_weights.pth')

        # weights = torch.load('model_weights.pth')
        # self.model.load_state_dict(weights)



### ------- You can make changes to this file from above this line --------------

    # This is a custom Evaluation function. You should not change this function
    # You can add other methods, or other functions to perform evaluation for
    # yourself. However, this evalution function will be used to evaluate your model
    def EvaluateModel(self):
        attacked = 0
        reached_goal = 0
        state = game_state.GameState()
        for _ in range(100000):
            action = self.GetAction(state)
            obs = state.Update(action)
            if(obs==game_state.GameObservation.Enemy_Attacked):
                attacked += 1
            elif(obs==game_state.GameObservation.Reached_Goal):
                reached_goal += 1
        return (attacked, reached_goal)