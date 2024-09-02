import numpy as np
import events as e
import settings as s

from collections import deque

from typing import List

import wandb
from .config import *

from .rulebased_teacher import TeacherModel

# Constants
# events
COIN_CLOSE = 'COIN_CLOSE'
COIN_CLOSER = 'COIN_CLOSER'
BOMB_TIME1 = 'BOMB_TIME1' # 1 step to explode
BOMB_TIME2 = 'BOMB_TIME2' # 2 steps to explode
BOMB_TIME3 = 'BOMB_TIME3' # 3 steps to explode
BOMB_TIME4 = 'BOMB_TIME4' # 4 steps to explode
BOMB_DROPPED_FOR_CRATE = 'BOMB_DROPPED_FOR_CRATE' # Crates will be destroyed by the dropped bomb
EXCAPE_FROM_BOMB = 'EXCAPE_FROM_BOMB'
LOOP_DETECTED = 'LOOP_DETECTED'
NEW_CELL_FOUND = 'NEW_CELL_FOUND' # The agent found a new cell


def setup_training(self):
    self.visited_history = deque([], 5)
    self.coin_history = []
    self.episode = 0
    
    self.teacher = TeacherModel()

    # Compose events as config
    
    if WANDB:
        # Initialize wandb
        wandb.init(
            project="bomberman_rl",
            entity="your-entity-name",  # Replace with your wandb username or team name
            config={
                "model": MODEL_NAME + '_' + MODEL_TYPE,
                "learning_rate": self.model.optimizer.param_groups[0]['lr'],
                "gamma": self.model.gamma,
                "n_layers": self.model.n_layers,
                "seq_len": self.model.seq_len,
                "hidden_dim": self.model.hidden_dim,
            }
        )

        # Log model architecture
        wandb.watch(self.model)
    
    
    
    

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[dict]) -> None:
    self.logger.debug(f'Encountered game event(s) {", ".join([event for event in events])}')
    
    # Exploration reward: If the agent found a new cell, add an event to events list
    if new_game_state['self'][3] not in self.visited_history:
        events.append(NEW_CELL_FOUND)
    
    # add position to visited history
    self.visited_history.append(new_game_state['self'][3])
    # check if the agent is in a loop, if so, add an event to events list
    if self.visited_history.count(new_game_state['self'][3]) > 3:
        events.append(LOOP_DETECTED)
    
    # distance to coins: if getting close to coins at the first time, add an event to events list
    coins_pos = old_game_state['coins']
    
    for coin_pos in coins_pos:
        if np.linalg.norm(np.array(coin_pos) - np.array(new_game_state['self'][3])) < 4: # falls into the coin range
            if coin_pos not in self.coin_history:
                events.append(COIN_CLOSE)
                self.coin_history.append(coin_pos)
            if np.linalg.norm(np.array(coin_pos) - np.array(new_game_state['self'][3])) < np.linalg.norm(np.array(coin_pos) - np.array(old_game_state['self'][3])):
                events.append(COIN_CLOSER)
    
    # distance to bombs: if still in danger zone, add an event to events list
    bombs = old_game_state['bombs']
    bombs_time = np.ones((s.COLS, s.ROWS)) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bombs_time.shape[0]) and (0 < j < bombs_time.shape[1]):
                bombs_time[i, j] = min(bombs_time[i, j], t)
    if bombs_time[new_game_state['self'][3]] == 1:
        events.append(BOMB_TIME1)
    elif bombs_time[new_game_state['self'][3]] == 2:
        events.append(BOMB_TIME2)
    elif bombs_time[new_game_state['self'][3]] == 3:
        events.append(BOMB_TIME3)
    elif bombs_time[new_game_state['self'][3]] == 4:
        events.append(BOMB_TIME4)
    
    
    # If the agent was in danger zone but now safe, add an event to events list
    if bombs_time[old_game_state['self'][3]] < 5 and bombs_time[new_game_state['self'][3]] == 5:
        events.append('EXCAPE_FROM_BOMB')
        
    # If the agent dropped a bomb and crates will be destroyed, add an event to events list
    crates = old_game_state['field'] == 1
    if self_action == 'BOMB':
        for (i, j) in [(new_game_state['self'][3][0] + h, new_game_state['self'][3][1]) for h in range(-3, 4)] + [(new_game_state['self'][3][0], new_game_state['self'][3][1] + h) for h in range(-3, 4)]:
            if (0 < i < crates.shape[0]) and (0 < j < crates.shape[1]) and crates[i, j] and bombs_time[i, j] == 5:
                events.append(BOMB_DROPPED_FOR_CRATE)
                
    
    #self.logger.info(f'Events: ',{" ,"}.join([event for event in events]))
    self.logger.info(f'Events: {events}')
    self.model.rewards.append(reward_from_events(events))
                    
    

def end_of_round(self, last_game_state, last_action, events): 
    # record the last game state info
    self.model.rewards.append(reward_from_events(events))
    self.model.scores.append(last_game_state['self'][1])
    
    # update the model parameters
    self.model.train()
    
    # log the game info
    self.logger.info(f'Episode {self.model.episode} ended with score {last_game_state["self"][1]}')
    self.logger.info(f'Events: {events}')
    self.logger.info(f'Rewards: {self.model.final_rewards[-1]}')
    self.logger.info(f'Discounted rewards: {self.model.final_discounted_rewards[-1]}')
    self.logger.info(f'Scores: {self.model.scores[-1]}')
    
    # Log metrics to wandb
    if WANDB:
        wandb.log({
            "loss": self.model.loss_values[-1],
            "teacher_loss": self.model.teacher_loss[-1],
            "policy_loss": self.model.policy_loss[-1],
            "reward": self.model.final_rewards[-1],
            "discounted_reward": self.model.final_discounted_rewards[-1],
            "score": self.model.scores[-1]
        })
    
    # Reset the model
    self.model.reset()
    
    self.model.episode += 1
    
    # Save model for every 200 episodes
    if self.model.episode % 200 == 0:
        save_path = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '_'+ 
                          MODEL_TYPE + '_seq_' + str(SEQ_LEN) + '_layer_' + 
                          str(N_LAYERS) + '_' + str(self.model.episode) + '.pt')
        self.model.save(save_path)


def reward_from_events(events) -> float:
    reward = 0
    game_rewards = {
        e.INVALID_ACTION: -0.1,
        e.MOVED_LEFT: -0.01,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.03,
        e.BOMB_DROPPED: -0.01,
        
        LOOP_DETECTED: -0.1,
        
        e.CRATE_DESTROYED: 0.05,
        e.COIN_FOUND: 0.3,
        COIN_CLOSE: 0.03,
        COIN_CLOSER: 0.25,
        e.COIN_COLLECTED: 4,
        
        BOMB_TIME3: -0.2,
        BOMB_TIME2: -0.3,
        BOMB_TIME1: -0.5,
        EXCAPE_FROM_BOMB: 0.5,
        e.BOMB_EXPLODED: 0,
        e.OPPONENT_ELIMINATED: 0,
        
        NEW_CELL_FOUND: 0.2,
        
        BOMB_DROPPED_FOR_CRATE: 0.2,
        
        e.KILLED_OPPONENT: 5,
        e.GOT_KILLED: -10,
        e.KILLED_SELF: -5,
        e.SURVIVED_ROUND: 5
    }
    reward = sum([game_rewards[event] for event in events])
    
    return reward


# if __name__ == '__main__':
#     print('Training the agent...')

#     # Initialize wandb
#     wandb.init(project="MLE_Bomberman", name="991211lja")

#     # Configure wandb to log hyperparameters and metrics
#     wandb.config.update({
#         "model_name": MODEL_NAME,
#         "feature_dim": 22,
#         "action_dim": 6,
#         "hidden_dim": 128,
#         "gamma": 0.99
#     })

#     # Log metrics during training
#     def log_metrics(episode, reward, loss):
#         wandb.log({
#             "episode": episode,
#             "reward": reward,
#             "loss": loss,
#             "score": score,
#             "events": events
#         })

#     # Make sure to call log_metrics() after each episode in your training loop
    