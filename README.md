# bomberman_rl

Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Basic Setup of the game

1. The arena has 17 rows and 17 columns.
2. entites in the game include:
   - Player (User agent)
   - Components (Other agents)
   - Coins: some are visible by the player and some are unvisible in the crates
   - Crates: blocks that can be destroyed by bombs
   - Bombs: explodes with a range of 3 and kill the agents and crates in its range, after 4 episodes
   - Explosions
3. Agents are randomly positioned in one of the four corners of the arena.
4. The player can move up, down, left, or right, place a bomb and no action in 0.5 seconds, based on their observation of the environment.

## States representation

1. Features extraction (For value-based methods):
   

2. High dimensional state representation (For DRL): 


## Reward shaping




## Models




## Test

Test code for task 1-4 in Main.py, if you want to show, just delete "--no-gui" in the corresponding command line:
- Task I-I: `python main.py play --agents Yu_policy_agent peaceful_agent peaceful_agent peaceful_agent --train 1 --scenario coin-heaven --n-round 20000 --no-gui`
- Task I-II: `python main.py play --agents Yu_policy_agent rule_based_agent peaceful_agent peaceful_agent --train 1 --scenario coin-heaven --n-round 20000 --no-gui`
- Task I-III: `python main.py play --agents Yu_policy_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario coin-heaven --n-round 20000 --no-gui`
- Task II-I: `python main.py play --agents Yu_policy_agent peaceful_agent peaceful_agent peaceful_agent --train 1 --scenario loot-crate --n-round 20000 --no-gui`
- Task II-II: `python main.py play --agents Yu_policy_agent rule_based_agent peaceful_agent peaceful_agent --train 1 --scenario loot-crate --n-round 20000 --no-gui`
- Task II-III: `python main.py play --agents Yu_policy_agent rule_based_agent rule_based_agent rule_based_agent --train 1 --scenario loot-crate --n-round 20000 --no-gui`
- Task III-I: `python main.py play --agents Yu_policy_agent peaceful_agent peaceful_agent peaceful_agent --train 1 --scenario classic --n-round 20000 --no-gui`
- Task III-II: `python main.py play --agents Yu_policy_agent rule_based_agent peaceful_agent peaceful_agent --train 1 --scenario classic --n-round 20000 --no-gui`
- Task III-III: `python main.py play --no-gui --my-agent Yu_policy_agent --train 1 --scenario classic --n-round 20000 --no-gui`

