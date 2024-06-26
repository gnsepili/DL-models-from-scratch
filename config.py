import torch

BATCH_SIZE = 64
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "shakespeare_model.pth"
TOKENIZER_PATH = "tokenizer.pkl"