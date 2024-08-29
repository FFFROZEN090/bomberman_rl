"""
This file contains the utility functions for the DQN agent.
"""

import os
import torch
import numpy as np



"""
Converts the game state into a 17x17x14 numpy array representation.

Channel mapping:
0-3: Players, 4: Brick, 5: Box, 6: Coin, 7-10: Bomb countdown,
11: Blast area, 12: Explosion, 13: Empty

Args:
    game_state (dict): The current game state.

Returns:
    numpy.ndarray: A 17x17x14 array representing the game state.
"""
def get_low_level_state(game_state):
    # Initialize the state array
    state = np.zeros((17, 17, 14), dtype=np.int8)
    
    # Extract relevant information from game state
    field = game_state['field']
    self_pos = game_state['self'][3]
    others = game_state['others']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    
    # Fill in the state array
    for x in range(17):
        for y in range(17):
            if field[x, y] == -1:  # Wall
                state[x, y, 4] = 1
            elif field[x, y] == 1:  # Crate
                state[x, y, 5] = 1
            else:  # Free space
                state[x, y, 13] = 1

    # Add self position
    state[self_pos[0], self_pos[1], 0] = 1
    
    # Add other players
    for i, other in enumerate(others):
        if i < 3:  # We only have channels for up to 3 other players
            other_pos = other[3]
            state[other_pos[0], other_pos[1], i+1] = 1
    
    # Add coins
    for coin_pos in coins:
        state[coin_pos[0], coin_pos[1], 6] = 1
    
    # Add bombs and their countdown
    for bomb_pos, countdown in bombs:
        state[bomb_pos[0], bomb_pos[1], 13-countdown] = 1  # CD 4 to CD 1 are channels 7 to 10
        # Add bomb danger zones
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:  # Manhattan distance <= 2
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 17 and 0 <= ny < 17:  # Check boundaries
                        state[nx, ny, 13-countdown] = 1

    
    # Add explosion map
    state[:, :, 11] = np.where(explosion_map > 0, 1, 0)
    
    # Add actual explosions
    state[:, :, 12] = np.where(explosion_map == 1, 1, 0)
    
    return state

"""
Extract high level state from low level state.

Low Level Status:
Players [0-3]: [14] [17] [18]
Brick [4]: [14] [17] [19]
Box [5]: [14] [17] [19] [18]
Coin [6]: [15] [18]
Bomb Countdown [7-10]: [19] [20] [18]
Blast Area [11]: [20] [18]
Explosion [12]: [18]
Empty [13]: [16] [18]

High Level Status:
[14]: Blocked
[15]: Reward
[16]: Safe
[17]: Destroyable
[18]: Changeable
[19]: Unchangeable
[20] to [23]: Different level danger area

Policies:
1. Dead zone: Check blast areas and surrounding blocks
2. Reward path: Find nearby coins and calculate paths

Input: 17x17x14 low level state
Output: 17x17x8 high level state
"""
def get_high_level_state(state):
    # Create a 17x17x8 numpy array representation
    high_level_state = np.zeros((17, 17, 10), dtype=np.int8)

    # Set blocked areas (channel 0)
    high_level_state[:, :, 0] = np.any(state[:, :, :6] == 1, axis=2)

    # Set reward areas (channel 1)
    high_level_state = check_reward_zone(state, high_level_state)

    # TODO: Reconsider the safe area
    # Set safe areas (channel 2)
    high_level_state[:, :, 2] = state[:, :, 13]  # Empty spaces are safe

    # Set destroyable blocks (channel 3)
    high_level_state[:, :, 3] = np.any(state[:, :, [0, 1, 2, 3, 5]] == 1, axis=2)  # Players and boxes are destroyable

    # Set changeable blocks (channel 4)
    high_level_state[:, :, 4] = np.any(state[:, :, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]] == 1, axis=2)

    # Set unchangeable blocks (channel 5)
    high_level_state[:, :, 5] = state[:, :, 4]  # Stone walls are unchangeable
    high_level_state[:, :, 5] = np.where(high_level_state[:, :, 4] == 1, 0, high_level_state[:, :, 5])

    # Set different level danger area (channel 6 to 9)
    for i in range(6, 10):
        high_level_state = check_danger_area(state, high_level_state, i - 5)

    return high_level_state


"""
Danger area:
1. For player position, check the surrounding area
    1.1 If our target is find one-step away, check player's UP, DOWN, LEFT, RIGHT cells, if any of them is count down 1, set channel 6 to 1
    1.2 If our target is find two-step away, check player's 2 steps reachable cells, if any of them is count down 2, set channel 6 to 1
    1.3 If our target is find three-step away, check player's 3 steps reachable cells, if any of them is count down 3, set channel 6 to 1
    1.4 If our target is find four-step away, check player's 4 steps reachable cells, if any of them is count down 4, set channel 6 to 1
2. Reachable cells:
    2.1 For one-step away, check player's UP, DOWN, LEFT, RIGHT, LOCAL cells
    2.2 For two-step away, check player's One-step: UP, DOWN, LEFT, RIGHT, LOCAL, Second-step: UP, DOWN, LEFT, RIGHT, LOCAL based on one-step's result move result
    2.3 For three-step away, check player's One-step: UP, DOWN, LEFT, RIGHT, LOCAL, Second-step: UP, DOWN, LEFT, RIGHT, LOCAL, Third-step: UP, DOWN, LEFT, RIGHT, LOCAL based on one-step's result move result
    2.4 For four-step away, check player's One-step: UP, DOWN, LEFT, RIGHT, LOCAL, Second-step: UP, DOWN, LEFT, RIGHT, LOCAL, Third-step: UP, DOWN, LEFT, RIGHT, LOCAL, Fourth-step: UP, DOWN, LEFT, RIGHT, LOCAL based on one-step's result move result

Input: 
    state: A 17x17x14 state
    high_level_state: A 17x17x10 high level state
    step: The step of the player and corresponding bomb countdown, danger level

Return: A 17x17x10 high level state 
"""
def check_danger_area(state, high_level_state, step):
    # Initialize a 9x9 grid for recording reachable cells
    reachable = np.zeros((9, 9), dtype=np.int8)
    center = 4  # Center of the 9x9 grid
    reachable[center, center] = 1  # Mark the central cell as reachable

    # Define directions: Left, Right, Up, Down
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Expand reachable area for each step
    for _ in range(step):
        new_reachable = reachable.copy()
        for i in range(9):
            for j in range(9):
                if reachable[i, j] == 1:
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 9 and 0 <= nj < 9:
                            new_reachable[ni, nj] = 1
        reachable = new_reachable

    # Get player position
    player_pos = np.where(state[:, :, 0] == 1)
    if player_pos[0].size > 0 and player_pos[1].size > 0:
        player_x, player_y = player_pos[0][0], player_pos[1][0]

        # Translate reachable grid to game state coordinates
        for i in range(9):
            for j in range(9):
                if reachable[i, j] == 1:
                    x = player_x + (i - center)
                    y = player_y + (j - center)
                    if 0 <= x < 17 and 0 <= y < 17:
                        # Reset reachable cell to 0 if there's a brick
                        if state[x, y, 4] == 1:
                            reachable[i, j] = 0
                        else:
                            # Check for danger in reachable cells
                            bomb_countdown = state[x, y, 7:11]  # Channels 7-10 for bomb countdown
                            if any(bomb_countdown == step):
                                high_level_state[x, y, 6 - 1 + step] = 1  # Mark as dangerous in channel 6

    return high_level_state
        

"""
Find path from current position to the reward zone
Input: A 17x17x14 state
        Wall: 4
        Box: 5
        Coin: 6
        Empty: 13
        Agent: 0 or 1 or 2 or 3
        Reward: 15

Return: Path length from Agent to Reward

From agent position, find the path to the reward zone with Dijkstra's algorithm

"""
def check_reward_zone(state, high_level_state):
    # For player 0, find 7x7 area, if there is a coin, set channel 15 to 1
    player_pos = np.where(state[:, :, 0] == 1)
    if player_pos[0].size > 0 and player_pos[1].size > 0:
        player_x, player_y = player_pos[0][0], player_pos[1][0]
        x_start, x_end = max(0, player_x - 7//2), min(17, player_x + 7//2 + 1)
        y_start, y_end = max(0, player_y - 7//2), min(17, player_y + 7//2 + 1)
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if state[x, y, 6] == 1:
                    high_level_state[x, y, 1] = 1


    return high_level_states



def get_state(game_state):
    # Initialize the state
    state = np.zeros((17, 17, 24), dtype=np.int8)
    low_level_state = get_low_level_state(game_state)
    high_level_state = get_high_level_state(low_level_state)

    # Concatenate the low level state and high level state
    state[:, :, :14] = low_level_state
    state[:, :, 14:] = high_level_state
    return state

