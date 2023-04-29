def get_hyperparams():
    # BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS 256
    BATCH_SIZE = 48  # 48
    # BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
    IMAGE_SIZE = (15, 15, 15)
    # how many frames to keep
    # in other words, how many observations the network can see
    FRAME_HISTORY = 4  # default 4
    # the frequency of updating the target network
    UPDATE_FREQ = 4  # default 4
    # DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
    GAMMA = 0.9  # 0.99
    # REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
    MEMORY_SIZE = 1e5  # 6#3   # to debug on bedale use 1e4
    # consume at least 1e6 * 27 * 27 * 27 bytes
    INIT_MEMORY_SIZE = MEMORY_SIZE // 20  # 5e4
    # each epoch is 100k played frames
    STEPS_PER_EPOCH = 15000 // UPDATE_FREQ  # default prev: 15000, diana: 100000 e.g. 100000//4

    # TODO: understand: EPOCHS_PER_EVAL, EVAL_EPISODE
    # TODO: consider global history length setting from Medical Player

    # num training epochs in between model evaluations
    EPOCHS_PER_EVAL = 1  # default: 1
    # the number of episodes to run during evaluation
    EVAL_EPISODE = 20  # default: 50
    # the number of epochs to run during training
    EPOCHS = 20  # default: 20
    # the number of steps to perform when evaluating
    STEPS_PER_EVAL = 20  # default: 50
    # maximum number of movements per step
    MAX_NUM_FRAMES = 1500  # 10000 # default: 1500
    # how many epochs should be saved?
    MAX_TO_KEEP = EPOCHS  # default: EPOCHS = 20
    # use multiscaling?
    MULTISCALE = True  # default: True

    ATTENTION = 16  # 0 means no attention
    SEED = 1
    return BATCH_SIZE, IMAGE_SIZE, FRAME_HISTORY, UPDATE_FREQ, GAMMA, MEMORY_SIZE, INIT_MEMORY_SIZE, STEPS_PER_EPOCH, \
           EPOCHS_PER_EVAL, EVAL_EPISODE, EPOCHS, STEPS_PER_EVAL, MAX_NUM_FRAMES, MAX_TO_KEEP, MULTISCALE, ATTENTION, SEED