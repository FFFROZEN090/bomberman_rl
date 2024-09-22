"""
This file contains the utility functions for the DQN agent.
"""

import os
import torch
import numpy as np
from collections import deque
from random import shuffle

SEARCH_DEPTH = 8

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
        for dx in range(0, -4, -1):
            nx = bomb_pos[0] + dx
            if state[4, nx, bomb_pos[1]] == 1:
                break
            if 1 <= nx < 16:
                state[7, nx, bomb_pos[1]] = max(state[8, nx, bomb_pos[1]], countdown + 1)
        for dx in range(0, 4):
            nx = bomb_pos[0] + dx
            if state[4, nx, bomb_pos[1]] == 1:
                break
            if 1 <= nx < 16:
                state[7, nx, bomb_pos[1]] = max(state[8, nx, bomb_pos[1]], countdown + 1)
        for dy in range(0, -4, -1):
            ny = bomb_pos[1] + dy
            if state[4, bomb_pos[0], ny] == 1:
                break
            if 1 <= ny < 16:
                state[7, bomb_pos[0], ny] = max(state[8, bomb_pos[0], ny], countdown + 1)
        for dy in range(0, 4):
            ny = bomb_pos[1] + dy
            if state[4, bomb_pos[0], ny] == 1:
                break
            if 1 <= ny < 16:
                state[7, bomb_pos[0], ny] = max(state[8, bomb_pos[0], ny], countdown + 1)
    # Add explosion map, assigned the explosion map with corresponding value
    state[8, :, :] = explosion_map

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
    high_level_state = np.zeros((6, 17, 17), dtype=np.int8)

    # Blocked areas (channel 0)
    high_level_state[0, :, :] = np.any(state[[4, 5, 8], :, :] >= 1, axis=0)  # Other players, brick walls, boxes
    high_level_state[0, :, :] = np.where(state[7, :, :] == 1, 1, high_level_state[0, :, :])  # Bomb countdown areas are blocked

    # Safe areas (channel 1) : check the cell is empty and channel 7 is 0
    high_level_state[1, :, :] = np.where(state[9, :, :] == 1, 1, 0)  # Empty cells are safe
    high_level_state[1, :, :] = np.where(state[7, :, :] == 2, 0, high_level_state[1, :, :])  # Bomb countdown cells are not safe
    high_level_state[1, :, :] = np.where(state[7, :, :] == 1, 0, high_level_state[1, :, :])

    # Blast area is not safe
    high_level_state[1, :, :] = np.where(state[8, :, :] >= 1, 3, high_level_state[1, :, :])  # Blast area cells are not safe

    # Destroyable blocks (channel 2)
    high_level_state[2, :, :] = np.any(state[[1,2,3,5], :, :] == 1, axis=0)  # Players are destroyable

    # Coin Target areas (channel 3)
    high_level_state[3, :, :] = coin_target(state)

    # Safe spots areas (channel 4)
    high_level_state[4, :, :] = safe_spots(state)

    # Bomb spot: Near the crate, suitable for bomb, calculate nearby crate, the value is the number of crates
    high_level_state[5, :, :] = bomb_spot(state)

    return high_level_state

def bomb_spot(state):
    # Find agent's position (player 0)
    player_pos = np.where(state[0, :, :] == 1)
    agent_x, agent_y = player_pos[0][0], player_pos[1][0]

    # Define 7x7 grid boundaries around the agent
    x_min = max(1, agent_x - SEARCH_DEPTH)
    x_max = min(15, agent_x + SEARCH_DEPTH)
    y_min = max(1, agent_y - SEARCH_DEPTH)
    y_max = min(15, agent_y + SEARCH_DEPTH)

    # Define blocked positions
    # Positions are blocked if they are:
    # - Brick walls (state[4] == 1)
    # - Boxes (state[5] == 1)
    blocked = (
        (state[4, :, :] == 1) |  # Brick walls
        (state[5, :, :] == 1)    # Boxes
    )

    # Initialize bomb spot matrix with zeros
    bomb_spot = np.zeros((17, 17), dtype=np.int8)

    # Count the number of crates in the 7x7 grid
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if not blocked[x, y]:
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    neighbor_x = x + dx
                    neighbor_y = y + dy
                    if x_min <= neighbor_x <= x_max and y_min <= neighbor_y <= y_max:
                        if state[5, neighbor_x, neighbor_y] == 1:
                            bomb_spot[x, y] += 1

    return bomb_spot

def distance_to_empty_cells(state):
    import numpy as np
    from collections import deque

    # Find agent's position (player 0)
    player_pos = np.where(state[0, :, :] == 1)
    agent_x, agent_y = player_pos[0][0], player_pos[1][0]

    # Define 7x7 grid boundaries around the agent
    x_min = max(1, agent_x - SEARCH_DEPTH)
    x_max = min(15, agent_x + SEARCH_DEPTH)
    y_min = max(1, agent_y - SEARCH_DEPTH)
    y_max = min(15, agent_y + SEARCH_DEPTH)

    # Define empty cells within the 7x7 grid
    # Empty cells are positions where there is:
    # - No brick wall (state[4] == 0)
    # - No box (state[5] == 0)
    # - No bomb countdown (state[7] == 0)
    # - No blast area (state[8] == 0)
    # Note: Other players are ignored when identifying empty cells
    empty_cells = np.zeros((17, 17), dtype=bool)
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if (
                (state[4, x, y] == 0) and  # Not a brick wall
                (state[5, x, y] == 0) and  # Not a box
                (state[7, x, y] == 0) and  # No bomb countdown
                (state[8, x, y] == 0)      # Not a blast area
            ):
                empty_cells[x, y] = True

    # Initialize distance matrix with infinity
    distance = np.full((17, 17), np.inf)
    distance[agent_x, agent_y] = 0

    # Define blocked positions for BFS traversal
    # Positions are blocked if they are:
    # - Brick walls (state[4] == 1)
    # - Boxes (state[5] == 1)
    # - Other players (state[1], state[2], state[3] == 1)
    # Note: Bomb countdown areas are not blocked
    blocked = (
        (state[4, :, :] == 1) |  # Brick walls
        (state[5, :, :] == 1)

    )

    # Initialize BFS queue
    queue = deque()
    queue.append((agent_x, agent_y))

    # Perform BFS to compute distances to empty cells
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

    # Convert distances to integers and fill the distance matrix
    # Only for empty cells
    distance_int = np.zeros((17, 17), dtype=np.int8)
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if empty_cells[x, y] and np.isfinite(distance[x, y]):
                distance_int[x, y] = int(distance[x, y])

    return distance_int

""" 
Around 7x7 area of our player, find the safe spots.
The safe spots are the empty cells without any danger (bomb countdown, explosion, other players, brick walls, boxes).
The distance to the safe spots is calculated using Dijkstra's algorithm.
"""

def safe_spots(state):

    # Find agent's position (player 0)
    player_pos = np.where(state[0, :, :] == 1)
    agent_x, agent_y = player_pos[0][0], player_pos[1][0]

    # Define grid boundaries around the agent
    x_min = max(1, agent_x - SEARCH_DEPTH)
    x_max = min(15, agent_x + SEARCH_DEPTH)
    y_min = max(1, agent_y - SEARCH_DEPTH)
    y_max = min(15, agent_y + SEARCH_DEPTH)

    # Initialize distance matrix with -1 (unreachable)
    distance_int = np.full((17, 17), -1, dtype=np.int8)

    # Define blocked cells (brick walls and boxes)
    blocked = (state[4, :, :] == 1) | (state[5, :, :] == 1)
    distance_int[blocked] = -2  # Blocked cells

    # Define unsafe cells according to your safe area definition
    unsafe = (
        (state[9, :, :] != 1) |  # Not empty cells are unsafe
        (state[7, :, :] == 1) |  # Bomb countdown cells are unsafe
        (state[7, :, :] == 2) |
        (state[8, :, :] >= 1)    # Blast area cells are unsafe
    )
    unsafe &= ~blocked  # Exclude already blocked cells
    distance_int[unsafe] = -3  # Unsafe cells

    # Initialize BFS queue and set agent's position
    distance_int[agent_x, agent_y] = 0
    queue = deque()
    queue.append((agent_x, agent_y))

    # Perform BFS to compute distances to safe spots
    while queue:
        current_x, current_y = queue.popleft()
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor_x = current_x + dx
            neighbor_y = current_y + dy
            if x_min <= neighbor_x <= x_max and y_min <= neighbor_y <= y_max:
                # Only proceed if cell is unvisited, not blocked, and not unsafe
                if distance_int[neighbor_x, neighbor_y] == -1:
                    if not blocked[neighbor_x, neighbor_y] and not unsafe[neighbor_x, neighbor_y]:
                        distance_int[neighbor_x, neighbor_y] = distance_int[current_x, current_y] + 1
                        queue.append((neighbor_x, neighbor_y))

    return distance_int


""" 
If there is a coin in the 7x7 block, find the distance to the coin from the agent position using Dijkstra's algorithm.
The distance matrix record the distance to closest coin from agent position. Each cell value is the distance to the coin.
"""
def coin_target(state):
    # Define SEARCH_DEPTH
    SEARCH_DEPTH = 3  # Since the grid is 7x7, half of 7 minus 1

    # Find agent's position (player 0)
    player_pos = np.where(state[0, :, :] == 1)
    agent_x, agent_y = player_pos[0][0], player_pos[1][0]

    # Find coin positions
    coin_pos = np.where(state[6, :, :] == 1)

    # If there is no coin, return a matrix indicating unreachable (-1)
    if coin_pos[0].size == 0:
        return np.full((17, 17), -1, dtype=np.int8)

    # Define grid boundaries around the agent
    x_min = max(1, agent_x - SEARCH_DEPTH)
    x_max = min(15, agent_x + SEARCH_DEPTH)
    y_min = max(1, agent_y - SEARCH_DEPTH)
    y_max = min(15, agent_y + SEARCH_DEPTH)

    # Initialize distance matrix with -1 (unreachable)
    distance_int = np.full((17, 17), -1, dtype=np.int8)

    # Define blocked positions (bricks and boxes)
    blocked = (state[4, :, :] == 1) | (state[5, :, :] == 1)

    # Set blocked cells to -2 in distance_int
    distance_int[blocked] = -2

    # Initialize BFS queue and set agent's position
    distance_int[agent_x, agent_y] = 0
    queue = deque()
    queue.append((agent_x, agent_y))

    # Perform BFS to compute distances
    while queue:
        current_x, current_y = queue.popleft()
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor_x = current_x + dx
            neighbor_y = current_y + dy
            if x_min <= neighbor_x <= x_max and y_min <= neighbor_y <= y_max:
                if distance_int[neighbor_x, neighbor_y] == -1:  # Not visited and not blocked
                    if not blocked[neighbor_x, neighbor_y]:
                        distance_int[neighbor_x, neighbor_y] = distance_int[current_x, current_y] + 1
                        queue.append((neighbor_x, neighbor_y))

    return distance_int

def get_state(game_state, rotate, bomb_valid=False):
    # Initialize the state
    state = np.zeros((18, 17, 17), dtype=np.int8)
    low_level_state = get_low_level_state(game_state, rotate)
    high_level_state = get_high_level_state(low_level_state)
    # 

    # Append bomb valid to high level state
    bomb_valid = np.zeros((17, 17), dtype=np.int8)
    bomb_valid[:, :] = bomb_valid
    high_level_state = np.append(high_level_state, bomb_valid[np.newaxis, :, :], axis=0)
    return high_level_state

def rule_based_action(agent, game_state):
    """
    Called each game step to determine the agent's next action.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.
    """
    agent.logger.info('Picking action according to rule set')
    # Check if we are in a different round
    if game_state["round"] != agent.current_round:
        reset_self(agent)
        agent.current_round = game_state["round"]
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # If agent has been in the same location three times recently, it's a loop
    if agent.coordinate_history.count((x, y)) > 2:
        agent.ignore_others_timer = 5
    else:
        agent.ignore_others_timer -= 1
    agent.coordinate_history.append((x, y))

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    if (bombs_left > 0) and (x, y) not in agent.bomb_history: valid_actions.append('BOMB')
    agent.logger.debug(f'Valid actions: {valid_actions}')

    # Collect basic action proposals in a queue
    # Later on, the last added action that is also valid will be chosen
    action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    shuffle(action_ideas)

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates
    # Add other agents as targets if in hunting mode or no crates/coins left
    if agent.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
        targets.extend(others)

    # Exclude targets that are currently occupied by a bomb
    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]

    # Take a step towards the most immediately interesting target
    free_space = arena == 0
    if agent.ignore_others_timer > 0:
        for o in others:
            free_space[o] = False
    d = look_for_targets(free_space, (x, y), targets, agent.logger)
    if d == (x, y - 1): action_ideas.append('UP')
    if d == (x, y + 1): action_ideas.append('DOWN')
    if d == (x - 1, y): action_ideas.append('LEFT')
    if d == (x + 1, y): action_ideas.append('RIGHT')
    if d is None:
        agent.logger.debug('All targets gone, nothing to do anymore')
        action_ideas.append('WAIT')

    # Add proposal to drop a bomb if at dead end
    if (x, y) in dead_ends:
        action_ideas.append('BOMB')
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            action_ideas.append('BOMB')
    # Add proposal to drop a bomb if arrived at target and touching crate
    if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
        action_ideas.append('BOMB')

    # Add proposal to run away from any nearby bomb about to blow
    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas.extend(action_ideas[:4])

    # Pick last action added to the proposals list that is also valid
    while len(action_ideas) > 0:
        a = action_ideas.pop()
        if a in valid_actions:
            # Keep track of chosen action for cycle detection
            if a == 'BOMB':
                agent.bomb_history.append((x, y))

            return a



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
    

