MAX_EPOCHS = 100
TOLERANCE = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
NUM_CLASSES = 2
DEFAULT_SEED = 12
MAX_LEN = 1024
ACCUM_ITERS = 8
NUM_GPUS = 8

def set_constants(
    batch_size=BATCH_SIZE,
    accum_iters=ACCUM_ITERS,
    max_epochs=MAX_EPOCHS,
    tolerance=TOLERANCE,
    learning_rate=LEARNING_RATE,
):
    global BATCH_SIZE, ACCUM_ITERS, MAX_EPOCHS, TOLERANCE, LEARNING_RATE
    BATCH_SIZE = batch_size
    ACCUM_ITERS = accum_iters
    MAX_EPOCHS = max_epochs
    TOLERANCE = tolerance
    LEARNING_RATE = learning_rate