# Hyperparameters
MODEL_NAME = 'coin' # Model name
N_LAYERS = 2 # Number of layers for FF or LSTM
MODEL_TYPE = 'LSTM' # 'FF' or 'LSTM' or 'PPO'
SEQ_LEN = 4 # Sequence length for LSTM
LAST_EPISODE = 0 # Last episode number
TEACH_EPISODE = 1000 # Episode to start training
# Whether to use wandb
WANDB = True