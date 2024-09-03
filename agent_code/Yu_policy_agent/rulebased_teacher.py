import numpy as np
from collections import deque
from random import shuffle

class TeacherModel:
    def __init__(self):
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        self.ignore_others_timer = 0
        self.current_round = 0

    def act(self, game_state):
        """
        Determine the agent's next action based on the current game state.
        Returns a tuple containing the chosen action and additional information for training.
        """
        # Gather information about the game state
        arena = game_state['field']
        _, score, bombs_left, (x, y) = game_state['self']
        bombs = game_state['bombs']
        bomb_xys = [xy for (xy, t) in bombs]
        others = [xy for (n, s, b, xy) in game_state['others']]
        coins = game_state['coins']
        bomb_map = self._get_bomb_map(arena, bombs)

        # Update agent's state
        self._update_agent_state(x, y)

        # Get valid actions
        valid_actions = self._get_valid_actions(arena, game_state, bomb_map, bombs_left, x, y, others, bomb_xys)

        # Compile targets
        targets = self._compile_targets(arena, coins, others)

        # Calculate features for each action
        action_features = {}
        for action in valid_actions:
            features = self._calculate_features(game_state, action, targets, bomb_map)
            action_features[action] = features

        # Choose the best action based on features
        chosen_action = self._choose_action(action_features)

        # Update bomb history
        if chosen_action == 'BOMB':
            self.bomb_history.append((x, y))

        return chosen_action, {
            'valid_actions': valid_actions,
            'action_features': action_features,
        }

    def _get_bomb_map(self, arena, bombs):
        bomb_map = np.ones(arena.shape) * 5
        for (xb, yb), t in bombs:
            for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
                if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                    bomb_map[i, j] = min(bomb_map[i, j], t)
        return bomb_map

    def _update_agent_state(self, x, y):
        if self.coordinate_history.count((x, y)) > 2:
            self.ignore_others_timer = 5
        else:
            self.ignore_others_timer -= 1
        self.coordinate_history.append((x, y))

    def _get_valid_actions(self, arena, game_state, bomb_map, bombs_left, x, y, others, bomb_xys):
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
        if (bombs_left > 0) and (x, y) not in self.bomb_history: valid_actions.append('BOMB')
        return valid_actions

    def _compile_targets(self, arena, coins, others):
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                     and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates
        if self.ignore_others_timer <= 0 or (len(crates) + len(coins) == 0):
            targets.extend(others)
        return targets

    def _calculate_features(self, game_state, action, targets, bomb_map):
        _, _, _, (x, y) = game_state['self']
        arena = game_state['field']
        
        if action == 'UP': new_pos = (x, y - 1)
        elif action == 'DOWN': new_pos = (x, y + 1)
        elif action == 'LEFT': new_pos = (x - 1, y)
        elif action == 'RIGHT': new_pos = (x + 1, y)
        else: new_pos = (x, y)
        
        features = {
            'distance_to_nearest_target': min(abs(new_pos[0] - t[0]) + abs(new_pos[1] - t[1]) for t in targets) if targets else 0,
            'bomb_danger': bomb_map[new_pos],
            'crate_nearby': int(any(arena[new_pos[0] + dx, new_pos[1] + dy] == 1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)])),
            'is_movement': int(action != 'WAIT' and action != 'BOMB'),
        }
        return features

    def _choose_action(self, action_features):
        best_action = None
        best_score = float('-inf')
        
        for action, features in action_features.items():
            score = (
                -5 * features['distance_to_nearest_target'] +
                -10 * features['bomb_danger'] +
                2 * features['crate_nearby'] +
                1 * features['is_movement']
            )
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def reset(self):
        self.bomb_history = deque([], 5)
        self.coordinate_history = deque([], 20)
        self.ignore_others_timer = 0