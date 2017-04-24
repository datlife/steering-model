# -------------------------------------------------------------------
# Parameter Setup
# -------------------------------------------------------------------
CURRENT_VERSION = 'v4/'
# Define constants
SEQ_LEN = 20
BATCH_SIZE = 4
LEFT_CONTEXT = 5

# Input image parameters
HEIGHT = 66
WIDTH = 200
CHANNELS = 3 # RGB

# LSTM paramters for keeping model state
RNN_SIZE = 32
RNN_PROJ = 32

# Output dimension setup
CSV_HEADER = "center,left,right,steering,throttle,brake,speed".split(",")
OUTPUTS = CSV_HEADER[3:6]  # steering, throttle, brake
OUTPUT_DIM = len(OUTPUTS)  # predict: steering angle, throttle and brake

# TRAINING
LEARN_RATE = 1e-4
KEEP_PROP = 1.0
AUX_COST_WEIGHT = 0.1  # wth is this?
KEEP_PROB_TRAIN = 0.75
