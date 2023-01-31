MAX_EPOCHS = 100
TOLERANCE = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_CLASSES = 2
DEFAULT_SEED = 12
MAX_LEN = 1024
ACCUM_ITERS = 8
NUM_GPUS = 8

def set_batch_size(batch_size, accum_iters=ACCUM_ITERS):
    global BATCH_SIZE, ACCUM_ITERS
    BATCH_SIZE = batch_size
    ACCUM_ITERS = accum_iters