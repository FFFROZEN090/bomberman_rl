# Hyperparameters
import os

MODEL_NAME = 'loot_crate_safe_bomb_reward' # Model name
LOAD_MODEL_NAME = 'loot_crate_safe_bomb_reward' # Model name to load
MODEL_TYPE = 'FF' # 'FF' or 'SFF' or 'LSTM' or 'PPO'
SEQ_LEN = 1 # Sequence length for LSTM
N_LAYERS = 2 # Number of layers for FF or LSTM
HIDDEN_DIM = 256 # Hidden dimension for FF or LSTM
LAST_EPISODE = 0 # Last episode number
ALPHA = 0 # Weight for imitation learning loss

FEATURE_DIM = 34

# Path

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', LOAD_MODEL_NAME + '_'+ 
                      MODEL_TYPE + '_seq_' + str(SEQ_LEN) + '_layer_' + 
                      str(N_LAYERS) + '_alpha_' + 
                      str(ALPHA) + '_hidden_' + str(HIDDEN_DIM) + '_' + str(LAST_EPISODE) + '.pt')


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

TEST_MODE = True # Whether to test the model

