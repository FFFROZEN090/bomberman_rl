"""
This file contains the utility functions for the DQN agent.
"""

import os
import torch
import numpy as np
from collections import deque

"""
Converts the game state into a 17x17x14 numpy array representation.

Channel mapping:
0-3: Players, 4: Brick, 5: Box, 6: Coin, 7: Bomb countdown,
8: Blast area, 9: Empty

Args:
    game_state (dict): The current game state.

Returns:
    numpy.ndarray: A 10x17x17 array representing the game state.
"""
def get_low_level_state(game_state, rotate=0):
    # Initialize the state array
    state = np.zeros((10, 17, 17), dtype=np.int8)
    
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
                state[4, x, y] = 1
            elif field[x, y] == 1:  # Crate
                state[5, x, y] = 1
            else:  # Free space
                state[9, x, y] = 1

    # Add self position
    state[0, self_pos[0], self_pos[1]] = 1
    
    # Add other players
    for i, other in enumerate(others):
        if i < 3:  # We only have channels for up to 3 other players
            other_pos = other[3]
            state[i+1, other_pos[0], other_pos[1]] = 1
    
    # Add coins
    for coin_pos in coins:
        state[6, coin_pos[0], coin_pos[1]] = 1
    
    # Add bombs and their countdown
    for bomb_pos, countdown in bombs:
        state[7, bomb_pos[0], bomb_pos[1]] = countdown + 1
        # Add bomb danger zones
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) <= 2:  # Manhattan distance <= 2
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 17 and 0 <= ny < 17:  # Check boundaries
                        state[7, nx, ny] = max(state[7, nx, ny], countdown + 1)


    # Add explosion map
    state[8, :, :] = np.where(explosion_map == 1, 1, 0)

    # Rotate the state if necessary
    state = rotate_state(state.copy(), rotate)
    
    return state

"""
Extract high level state from low level state.

Low level state:
0: Our player,
1: Other player 1,
2: Other player 2,
3: Other player 3, 
4: Brick, 
5: Box, 
6: Coin, 
7: Bomb countdown,
8: Blast area, 
9: Empty

High level state:
0: Blocked areas [Other players, brick walls, boxes], Derived from: [1, 2, 3, 4, 5],
1: Safe areas [Empty cells without any danger], Derived from: [7, 9],
2: Destroyable blocks [Players, boxes], Derived from: [1, 2, 3, 5],
3: Coin Target areas [Distance to the coin], Derived from: [0, 4, 5, 6],
4: Safe spots areas [Distance to the safe spots around 5x5 block of the Our player], Derived from: [0, 7, 9],



Input: 10x17x17 low level state
Output: 5x17x17 high level state
"""
def get_high_level_state(state):
    # Create a 17x17x8 numpy array representation
    high_level_state = np.zeros((5, 17, 17), dtype=np.int8)

    # Blocked areas (channel 0)
    high_level_state[0, :, :] = np.any(state[[1, 2, 3, 4, 5], :, :] == 1, axis=0)  # Other players, brick walls, boxes

    # Safe areas (channel 1) : check the cell is empty and channel 7 is 0
    high_level_state[1, :, :] = np.where(state[9, :, :] == 1, 1, 0)  # Empty cells are safe
    high_level_state[1, :, :] = np.where(state[7, :, :] == 1, 0, high_level_state[1, :, :])  # Bomb countdown cells are not safe

    # Destroyable blocks (channel 2)
    high_level_state[2, :, :] = np.any(state[[1, 2, 3, 5], :, :] == 1, axis=0)  # Players are destroyable

    # Coin Target areas (channel 3)
    high_level_state[3, :, :] = coin_target(state)

    # Safe spots areas (channel 4)
    high_level_state[4, :, :] = safe_spots(state)

    return high_level_state

""" 
Around 7x7 area of our player, find the safe spots.
The safe spots are the empty cells without any danger (bomb countdown, explosion, other players, brick walls, boxes).
The distance to the safe spots is calculated using Dijkstra's algorithm.
"""
def safe_spots(state):
    # Find agent's position (player 0)
    player_pos = np.where(state[0, :, :] == 1)
    agent_x, agent_y = player_pos[0][0], player_pos[1][0]

    # Define 7x7 grid boundaries around the agent
    x_min = max(0, agent_x - 6)
    x_max = min(16, agent_x + 6)
    y_min = max(0, agent_y - 6)
    y_max = min(16, agent_y + 6)

    # Initialize distance matrix with infinity
    distance = np.full((17, 17), np.inf)
    distance[agent_x, agent_y] = 0

    # Define blocked positions
    # Positions are blocked if they are:
    # - Brick walls (state[4] == 1)
    # - Boxes (state[5] == 1)
    # - Blast areas (state[8] == 1)
    blocked = (
        (state[4, :, :] == 1) |  # Brick walls
        (state[5, :, :] == 1) |  # Boxes
        (state[8, :, :] == 1)    # Blast areas
    )

    # Initialize BFS queue with (position, arrival_time)
    queue = deque()
    queue.append((agent_x, agent_y, 0))

    # Keep track of visited positions
    visited = np.zeros((17, 17), dtype=bool)
    visited[agent_x, agent_y] = True

    # Perform BFS to compute distances to safe spots
    while queue:
        current_x, current_y, arrival_time = queue.popleft()
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor_x = current_x + dx
            neighbor_y = current_y + dy
            if x_min <= neighbor_x <= x_max and y_min <= neighbor_y <= y_max:
                if not blocked[neighbor_x, neighbor_y]:
                    if not visited[neighbor_x, neighbor_y]:
                        # Check bomb danger area
                        bomb_countdown = state[7, neighbor_x, neighbor_y]
                        if bomb_countdown > 0:
                            # Bomb will explode in (bomb_countdown - 1) steps
                            bomb_explosion_time = bomb_countdown - 1
                            expected_arrival_time = arrival_time + 1
                            if expected_arrival_time >= bomb_explosion_time:
                                # Cannot proceed to this cell
                                continue
                        # Safe to proceed
                        distance[neighbor_x, neighbor_y] = arrival_time + 1
                        visited[neighbor_x, neighbor_y] = True
                        queue.append((neighbor_x, neighbor_y, arrival_time + 1))

    # Identify safe spots within the 7x7 grid
    safe_spots_mask = (~blocked) & (np.isfinite(distance))

    # Convert distances to integers
    distance_int = np.zeros((17, 17), dtype=np.int8)
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if safe_spots_mask[x, y]:
                distance_int[x, y] = int(distance[x, y])

    return distance_int


""" 
If there is a coin in the 7x7 block, find the distance to the coin from the agent position using Dijkstra's algorithm.
The distance matrix record the distance to closest coin from agent position. Each cell value is the distance to the coin.
"""
def coin_target(state):
    # Find agent's position (player 0)
    player_pos = np.where(state[0, :, :] == 1)
    agent_x, agent_y = player_pos[0][0], player_pos[1][0]
    
    # Find coin positions
    coin_pos = np.where(state[6, :, :] == 1)
    
    # If there is no coin, return a zero matrix
    if coin_pos[0].size == 0 or coin_pos[1].size == 0:
        return np.zeros((17, 17), dtype=np.int8)
    
    # Define 7x7 grid boundaries around the agent
    x_min = max(0, agent_x - 6)
    x_max = min(16, agent_x + 6)
    y_min = max(0, agent_y - 6)
    y_max = min(16, agent_y + 6)
    
    # Initialize distance matrix with infinity
    distance = np.full((17, 17), np.inf)
    distance[agent_x, agent_y] = 0
    
    # Define blocked positions (bricks and boxes)
    blocked = (state[4, :, :] == 1) | (state[5, :, :] == 1)
    
    # Initialize BFS queue
    queue = deque()
    queue.append((agent_x, agent_y))
    
    # Perform BFS to compute distances
    while queue:
        current_x, current_y = queue.popleft()
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor_x = current_x + dx
            neighbor_y = current_y + dy
            if x_min <= neighbor_x <= x_max and y_min <= neighbor_y <= y_max:
                if not blocked[neighbor_x, neighbor_y]:
                    if distance[neighbor_x, neighbor_y] > distance[current_x, current_y] + 1:
                        distance[neighbor_x, neighbor_y] = distance[current_x, current_y] + 1
                        queue.append((neighbor_x, neighbor_y))
    
    # Find coins within the 7x7 grid
    coins_in_grid = [
        (x, y) for x, y in zip(coin_pos[0], coin_pos[1])
        if x_min <= x <= x_max and y_min <= y <= y_max
    ]
    
    # If no coins are reachable within the grid, return zeros
    if not coins_in_grid:
        return np.zeros((17, 17), dtype=np.int8)
    
    # Convert distances to integers and fill the distance matrix
    distance_int = np.zeros((17, 17), dtype=np.int8)
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if np.isfinite(distance[x, y]):
                distance_int[x, y] = int(distance[x, y])
    
    return distance_int 

def get_state(game_state, rotate):
    # Initialize the state
    state = np.zeros((15, 17, 17), dtype=np.int8)
    low_level_state = get_low_level_state(game_state, rotate)
    high_level_state = get_high_level_state(low_level_state)

    # Concatenate the low level state and high level state
    state[:10, :, :] = low_level_state
    state[10:, :, :] = high_level_state
    return state


"""
Rotate the state by 90, 180, 270 degrees.
"""
def rotate_state(state, angle):
    if angle == 90:
        new_state = np.rot90(state, k=1, axes=(2, 1)).copy()
        return new_state
    elif angle == 180:
        new_state = np.rot90(state, k=2, axes=(2, 1)).copy()
        return new_state
    elif angle == 270:
        new_state = np.rot90(state, k=3, axes=(2, 1)).copy()
        return new_state
    else:
        return state.copy()
    

