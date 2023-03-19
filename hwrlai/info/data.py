from hwrlai.util.vec import Vec3

class Files:
    file_log = 'log'
    file_model = r'C:\\Users\\Skylar\\Dropbox\\rlbot\\data\\model.pkl'
    file_rw = 'rewards.csv'
    file_ts = 'timesteps.txt'
    file_cr = 'reward_cr.txt'
    file_pr = 'reward_pr.txt'
    file_ar = 'reward_ar.txt'
    file_gs = r'C:\\Users\\Skylar\\Dropbox\\rlbot\\states\\game\\game.state'
    file_ps = r'C:\\Users\\Skylar\\Dropbox\\rlbot\\states\\player\\player.state'
    
    # file_q_table = r'C:\\Users\\Skylar\\Dropbox\\rlbot\\data\\q_table.pickle'
    # file_qt_comp = file_q_table+'.bz2'
    # file_state_q = r'C:\\Users\\Skylar\\Dropbox\\rlbot\\data\\state_q.pickle'
    # file_sq_comp = file_state_q+'.bz2'

    file_nn_model = r'C:\\Users\\Skylar\\Dropbox\\rlbot\\data\\nn_model.h5'

class Field:
    MAX_SPEED = 2299.99
    FIELD_WIDTH = 8192
    FIELD_LENGTH = 10240
    FIELD_HEIGHT = 2044
    GOAL_WIDTH = 1900
    GOAL_HEIGHT = 640
    BALL_RADIUS = 92
    POS_MY_GOAL = Vec3(0, -5100, 18.34)
    POS_EN_GOAL = Vec3(0, 5100, 18.34)
    
    loc_boosts = (
        (0.0, -4240.0, 70.0),
        (-1792.0, -4184.0, 70.0),
        (1792.0, -4184.0, 70.0),
        (-3072.0, -4096.0, 73.0),
        (3072.0, -4096.0, 73.0),
        (- 940.0, -3308.0, 70.0),
        (940.0, -3308.0, 70.0),
        (0.0, -2816.0, 70.0),
        (-3584.0, -2484.0, 70.0),
        (3584.0, -2484.0, 70.0),
        (-1788.0, -2300.0, 70.0),
        (1788.0, -2300.0, 70.0),
        (-2048.0, -1036.0, 70.0),
        (0.0, -1024.0, 70.0),
        (2048.0, -1036.0, 70.0),
        (-3584.0, 0.0, 73.0),
        (-1024.0, 0.0, 70.0),
        (1024.0, 0.0, 70.0),
        (3584.0, 0.0, 73.0),
        (-2048.0, 1036.0, 70.0),
        (0.0, 1024.0, 70.0),
        (2048.0, 1036.0, 70.0),
        (-1788.0, 2300.0, 70.0),
        (1788.0, 2300.0, 70.0),
        (-3584.0, 2484.0, 70.0),
        (3584.0, 2484.0, 70.0),
        (0.0, 2816.0, 70.0),
        (- 940.0, 3310.0, 70.0),
        (940.0, 3308.0, 70.0),
        (-3072.0, 4096.0, 73.0),
        (3072.0, 4096.0, 73.0),
        (-1792.0, 4184.0, 70.0),
        (1792.0, 4184.0, 70.0),
        (0.0, 4240.0, 70.0),
    )    
    
