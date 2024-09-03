# Hyperparameters
import os

MODEL_NAME = 'Peaceful_agent' # Model name
MODEL_TYPE = 'LSTM' # 'FF' or 'LSTM' or 'PPO'
SEQ_LEN = 2 # Sequence length for LSTM
N_LAYERS = 3 # Number of layers for FF or LSTM
LAST_EPISODE = 0 # Last episode number
ALPHA = 1.0 # Weight for imitation learning loss

FEATURE_DIM = 22 # 22 for full dimension, 18 excluded dead ends and 14 excluded crates

# Path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', MODEL_NAME + '_'+ 
                          MODEL_TYPE + '_seq_' + str(SEQ_LEN) + '_layer_' + 
                          str(N_LAYERS) + '_feature_' + str(FEATURE_DIM) + '_alpha_' + 
                          str(ALPHA) + '_' + str(LAST_EPISODE) + '.pt')

WANDB = True # Whether to use wandb

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']