import common.game_constants as game_constants
import common.game_state as game_state
import pygame
import numpy as np

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

    def __init__(self) -> None:
        super(AIController, self).__init__()
        self.q_table = np.zeros((game_constants.GAME_WIDTH, game_constants.GAME_HEIGHT, len(game_state.GameActions)))
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1

    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        x, y = int(state.PlayerEntity.entity.x), int(state.PlayerEntity.entity.y)
        if np.random.uniform() < self.epsilon:
            # Random action
            action = np.random.choice(list(game_state.GameActions))
        else:
            # Choose action with highest Q-value
            action_values = self.q_table[x, y, :]
            action = game_state.GameActions(np.argmax(action_values))

        return action

    def TrainModel(self):
        epochs = 1000
        for _ in range(epochs):
            if(_ % 100 ==0):
                print("Epoch: "+ str(_))
                print()
            state = game_state.GameState()
            done = False
            while not done:
                # Choose action with epsilon-greedy policy
                action = self.GetAction(state)

                # Take action and observe next state and reward
                obs = state.Update(action)
                next_state = state
                reward = 0
                if obs == game_state.GameObservation.Enemy_Attacked:
                    reward = -200
                    done = True
                elif obs == game_state.GameObservation.Reached_Goal:
                    reward = 100
                    done = True
                elif obs == game_state.GameObservation.Nothing:
                    reward = -1

                # Update Q-value for current state-action pair
                x, y = int(state.PlayerEntity.entity.x), int(state.PlayerEntity.entity.y)
                next_action_values = self.q_table[int(next_state.PlayerEntity.entity.x), int(next_state.PlayerEntity.entity.y), :]
                td_error = reward + self.discount_factor * np.max(next_action_values) - self.q_table[x, y, action.value]
                self.q_table[x, y, action.value] += self.learning_rate * td_error

                # Transition to next state
                state = next_state



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