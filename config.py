# parameters for processing the dataset
DATA_PATH = 'data'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
OUTPUT_FILE = 'output_convo.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
EOS_ID = 3

TESTSET_SIZE = 25000

# model parameters

# [19530, 17449, 17585, 23444, 22884, 16435, 17085, 18291, 18931]
# BUCKETS = [(6, 8), (8, 10), (10, 12), (13, 15), (16, 19), (19, 22), (23, 26), (29, 32), (39, 44)]

# [37049, 33519, 30223, 33513, 37371]
# BUCKETS = [(8, 10), (12, 14), (16, 19), (23, 26), (39, 43)]

# BUCKETS = [(8, 10), (12, 14), (16, 19)]
BUCKETS = [(16, 19)]

NUM_LAYERS = 3
HIDDEN_SIZE = 512
EMBED_SIZE = 512
BATCH_SIZE = 128

LR = 0.0001
MAX_GRAD_NORM = 5.0


NUM_SAMPLES = 512
ENC_VOCAB = 24436
DEC_VOCAB = 24654
