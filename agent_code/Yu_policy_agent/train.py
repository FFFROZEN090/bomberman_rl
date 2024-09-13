import numpy as np
import events as e
import settings as s

from collections import deque

from typing import List

import wandb
from .config import *

from .rulebased_teacher import TeacherModel

if TEST_MODE:
    WANDB = False
    PRINT_INFO = True
else:
    WANDB = True
    PRINT_INFO = False

# Constants
# events
COIN_CLOSE = 'COIN_CLOSE'
COIN_CLOSER = 'COIN_CLOSER'
COIN_CLOSEST = 'COIN_CLOSEST'
BOMB_TIME1 = 'BOMB_TIME1' # 1 step to explode
BOMB_TIME2 = 'BOMB_TIME2' # 2 steps to explode
BOMB_TIME3 = 'BOMB_TIME3' # 3 steps to explode
BOMB_FARTHER = 'BOMB_FARTHER' # The agent is farther from the bomb
# BOMB_TIME4 = 'BOMB_TIME4' # 4 steps to explode
BOMB_DROPPED_AND_NO_SAFE_CELL = 'BOMB_DROPPED_AND_NO_SAFE_CELL' # The agent dropped a bomb and then there is no safe cell
BOMB_DROPPED_AT_DEAD_ENDS = 'BOMB_DROPPED_AT_DEAD_ENDS' # The agent dropped a bomb at dead ends
BOMB_DROPPED_FOR_CRATE = 'BOMB_DROPPED_FOR_CRATE' # Crates will be destroyed by the dropped bomb
BOMB_DROPPED_AT_NO_TARGET = 'BOMB_DROPPED_AT_NO_TARGET' # The agent dropped a bomb but no target is nearby
FALL_INTO_BOMB = 'FALL_INTO_BOMB' # The agent falls into the bomb range
WAIT_UNTIL_BOMB_EXPLODE = 'WAIT_UNTIL_BOMB_EXPLODE' # The agent waits until the bomb explodes
ESCAPE_FROM_BOMB = 'ESCAPE_FROM_BOMB'
ESCAPE_FROM_BOMB_BY_CORNER = 'ESCAPE_FROM_BOMB_BY_CORNER' # The agent escapes from the bomb by turning around a corner
LOOP_DETECTED = 'LOOP_DETECTED'
NEW_CELL_FOUND = 'NEW_CELL_FOUND' # The agent found a new cell
STAY_IN_SAFE_ZONE = 'STAY_IN_SAFE_ZONE' # The agent keeps in the safe zone



def setup_training(self):
    self.looped_detect_history = deque([], 14)
    self.newcell_detect_history = deque([], 10)
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
    if new_game_state['self'][3] not in self.newcell_detect_history:
        events.append(NEW_CELL_FOUND)
    
    # add position to visited history
    self.looped_detect_history.append(new_game_state['self'][3])
    self.newcell_detect_history.append(new_game_state['self'][3])
    # check if the agent is in a loop, if so, add an event to events list
    if self.looped_detect_history.count(new_game_state['self'][3]) >= 4:
        events.append(LOOP_DETECTED)
    
    # distance to coins: if getting close to coins at the first time, add an event to events list
    coins_pos = old_game_state['coins']
    
    for coin_pos in coins_pos:
        # if the manhattan distance is smaller than 4, then the agent is close to the coin
        if np.abs(np.array(coin_pos) - np.array(new_game_state['self'][3])).sum() < 4: # falls into the coin range
            if coin_pos not in self.coin_history:
                events.append(COIN_CLOSE)
                self.coin_history.append(coin_pos)
            if np.linalg.norm(np.array(coin_pos) - np.array(new_game_state['self'][3])) < np.linalg.norm(np.array(coin_pos) - np.array(old_game_state['self'][3])):
                events.append(COIN_CLOSER)
            if np.abs(np.array(coin_pos) - np.array(new_game_state['self'][3])).sum() < 2:
                events.append(COIN_CLOSEST)
    
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
    
    
    # If the agent dropped a bomb at the first step, add an event to events list
    if 'BOMB_DROPPED' in events and old_game_state['self'][3] in [(1,1), (1,s.ROWS-2), (s.COLS-2,1), (s.COLS-2,s.ROWS-2)]:
        events.append(BOMB_DROPPED_AND_NO_SAFE_CELL)
    
    # If the agent is farther from the bomb, add an event to events list
    if bombs_time[old_game_state['self'][3]] < 5:
        if bombs_time[new_game_state['self'][3]] == 5:
            # If the agent was in danger zone but now safe, add an event to events list
            events.append(ESCAPE_FROM_BOMB)
        for (xb, yb), t in bombs:
            if xb == old_game_state['self'][3][0] and abs(yb - old_game_state['self'][3][1]) < 4:
                # If the agent is farther from the bomb, add an event to events list
                if abs(yb - old_game_state['self'][3][1]) < abs(yb - new_game_state['self'][3][1]):
                    events.append(BOMB_FARTHER)
                # If the agent escapes from the bomb by turning around a corner, add an event to events list
                if xb != new_game_state['self'][3][0] and yb != new_game_state['self'][3][1]:
                    events.append(ESCAPE_FROM_BOMB_BY_CORNER)
            if yb == old_game_state['self'][3][1] and abs(xb - old_game_state['self'][3][0]) < 4:
                if abs(xb - old_game_state['self'][3][0]) < abs(xb - new_game_state['self'][3][0]):
                    events.append(BOMB_FARTHER)
                if xb != new_game_state['self'][3][0] and yb != new_game_state['self'][3][1]:
                    events.append(ESCAPE_FROM_BOMB_BY_CORNER)
    
    if bombs_time[new_game_state['self'][3]] == 5 and old_game_state['step'] >= 30:
        # If the agent is in a safe zone, add an event to events list
        events.append(STAY_IN_SAFE_ZONE)
        
    # If the agent is close to a bomb but fall in a safe cell, add an reward for waiting until the bomb explodes
    for (xb, yb), t in bombs:
        if np.abs(np.array(new_game_state['self'][3]) - np.array([xb, yb])).sum() < 4 and bombs_time[old_game_state['self'][3]] == 5 and 'WAITED' in events:
            events.append(WAIT_UNTIL_BOMB_EXPLODE)
        
    arena = new_game_state['field']
    # If the agent dropped a bomb at dead ends, add an event to events list
    dead_ends = [(i, j) for i in range(s.COLS) for j in range(s.ROWS) if (arena[i, j] == 0)
                    and ([arena[i + 1, j], arena[i - 1, j], arena[i, j + 1], arena[i, j - 1]].count(0) == 1)]
    if 'BOMB_DROPPED' in events and old_game_state['self'][2] > 0 and new_game_state['self'][3] in dead_ends:
        events.append(BOMB_DROPPED_AT_DEAD_ENDS)
    
    # If the agent falls into the bomb range from a safer zone, add an event to events list
    if self_action != 'BOMB' and bombs_time[old_game_state['self'][3]] == 5 and bombs_time[new_game_state['self'][3]] < 5:
        events.append(FALL_INTO_BOMB)
    
    # If the agent dropped a bomb and crates will be destroyed, add an event to events list
    crates = old_game_state['field'] == 1
    if 'BOMB_DROPPED' in events:
        for (i, j) in [(new_game_state['self'][3][0] + h, new_game_state['self'][3][1]) for h in range(-3, 4)] + [(new_game_state['self'][3][0], new_game_state['self'][3][1] + h) for h in range(-3, 4)]:
            if (0 < i < crates.shape[0]) and (0 < j < crates.shape[1]) and crates[i, j] and bombs_time[i, j] == 5:
                events.append(BOMB_DROPPED_FOR_CRATE)
        
        # add a bomb record
        self.model.bomb_dropped += 1
    
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
    self.logger.info(f'Survival time: {self.model.survival_time[-1]}')
    self.logger.info(f'Bomb dropped: {self.model.bomb_dropped_history[-1]}')
    self.logger.info(f'Scoring efficiency: {self.model.scoring_efficiency[-1]}')
    
    # Log metrics to wandb
    if WANDB:
        wandb.log({
            "loss": self.model.loss_values[-1],
            "teacher_loss": self.model.teacher_loss[-1],
            "policy_loss": self.model.policy_loss[-1],
            "reward": self.model.final_rewards[-1],
            "discounted_reward": self.model.final_discounted_rewards[-1],
            "score": self.model.scores[-1],
            "survival_time": self.model.survival_time[-1],
            "bomb_dropped": self.model.bomb_dropped_history[-1],
            "scoring_efficiency": self.model.scoring_efficiency[-1]
        })
    
    # Reset the model
    self.model.reset()
    
    self.model.episode += 1
    
    # Save model for every 5000 episodes
    if self.model.episode % 5000 == 0:
        save_path = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '_'+ 
                          MODEL_TYPE + '_seq_' + str(SEQ_LEN) + '_layer_' + 
                          str(N_LAYERS) + '_alpha_' + 
                          str(ALPHA) + '_hidden_' + str(HIDDEN_DIM) + '_'+ str(self.model.episode) + '.pt')
        self.model.save(save_path)
        
    # print(f'The model has {self.model.count_parameters():,} parameters')


def reward_from_events(events) -> float:
    reward = 0
    game_rewards = {
        e.INVALID_ACTION: -0.5,
        e.MOVED_LEFT: 0.1,
        e.MOVED_RIGHT: 0.1,
        e.MOVED_UP: 0.1,
        e.MOVED_DOWN: 0.1,
        e.WAITED: -0.2,
        e.BOMB_DROPPED: -0.1,
        
        LOOP_DETECTED: -0.3,
        
        e.CRATE_DESTROYED: 0.2,
        e.COIN_FOUND: 0.3,
        COIN_CLOSE: 0.15,
        COIN_CLOSER: 0.25,
        COIN_CLOSEST: 0.45,
        e.COIN_COLLECTED: 4,
        
        BOMB_TIME3: -0.2,
        BOMB_TIME2: -0.3,
        BOMB_TIME1: -0.5,
        FALL_INTO_BOMB: -0.8,
        ESCAPE_FROM_BOMB: 0.5,
        ESCAPE_FROM_BOMB_BY_CORNER: 0.5,
        WAIT_UNTIL_BOMB_EXPLODE: 0.5,
        STAY_IN_SAFE_ZONE: 0.05,
        BOMB_DROPPED_AND_NO_SAFE_CELL: -0.8,
        BOMB_FARTHER: 0.5,
        e.BOMB_EXPLODED: 0,
        e.OPPONENT_ELIMINATED: 0,
        
        NEW_CELL_FOUND: 0.1,
        
        BOMB_DROPPED_FOR_CRATE: 0.2,
        BOMB_DROPPED_AT_DEAD_ENDS: 0.5,
        
        e.KILLED_OPPONENT: 10,
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
    