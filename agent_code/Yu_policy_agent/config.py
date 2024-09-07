# Hyperparameters
import os

MODEL_NAME = 'loot_crate' # Model name
MODEL_TYPE = 'FF' # 'FF' or 'SFF' or 'LSTM' or 'PPO'
SEQ_LEN = 1 # Sequence length for LSTM
N_LAYERS = 2 # Number of layers for FF or LSTM
LAST_EPISODE = 0 # Last episode number
ALPHA = 0.2 # Weight for imitation learning loss

FEATURE_DIM = 30 # 30 for full dimension, 18 excluded dead ends and 14 excluded crates

# Path

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '_'+ 
                      MODEL_TYPE + '_seq_' + str(SEQ_LEN) + '_layer_' + 
                      str(N_LAYERS) + '_alpha_' + 
                      str(ALPHA) + '_' + str(LAST_EPISODE) + '.pt')


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

TEST_MODE = False # Whether to test the model

