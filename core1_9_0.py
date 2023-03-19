import os
import rlgym
import time

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import FaceBallReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward #incentivizes the agent to move towards the ball as quickly as possible

from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward # incentivizes the agent to move the ball towards the opponent's goal as quickly as possible

from rlgym.utils.reward_functions.common_rewards import BallYCoordinateReward
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward #encourage to stay close to ball

from rlgym.utils.reward_functions.common_rewards.misc_rewards import AlignBallGoal
from rlgym.utils.reward_functions.common_rewards.misc_rewards import VelocityReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward

#from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from stable_baselines3 import PPO

from hwrlai.helpers.functions import con_str_2_n_or_f
from hwrlai.helpers.functions import thetime
from hwrlai.helpers.functions import logit
from hwrlai.helpers.functions import prtit
from hwrlai.helpers.functions import terminate
#from hwrlai.helpers.functions import dR

from hwrlai.classes.actions import RLDiscreteAction
from hwrlai.classes.rewards import PenalizedReward
from hwrlai.classes.obs import CustomObsBuilder
from hwrlai.classes.terms import CustomTerminalCondition

from hwrlai.info.data import Files

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logit("")
logit("=====================================")
logit("starting...")

try:
    if terminate('RocketLeague.exe'):
        logit("train::: sleeping...")
        time.sleep(3)
except:
    pass

try:
    with open(Files.file_ts) as j:
        content = j.readlines()
        logit("train::: loaded settings successfully")
        
    settings = [x.strip() for x in content];
    
except:
    logit("train::: ERROR: getting settings failed or not found")
    settings_found = 0
    
    settings = []
    settings.append("ts=0")
    
#extracting settings
for line in settings:
    if not line.startswith("#"):
        line_split = line.split("=")
        fph = line_split[0]
        if line_split[0] == "ts" : 
            core_timesteps = con_str_2_n_or_f(line_split[1].strip())
            
if os.path.exists(Files.file_cr):            
    with open(Files.file_cr) as j:
        content = j.readlines()
        logit("previous cr: "+str(content[0]))
    
if os.path.exists(Files.file_pr):        
    with open(Files.file_pr) as j:
        content = j.readlines()
        logit("previous pr: "+str(content[0]))
        
if os.path.exists(Files.file_ar):        
    with open(Files.file_ar) as j:
        content = j.readlines()
        logit("previous ar: "+str(content[0]))         

if __name__ == '__main__':
    
    if not os.path.exists(Files.file_rw):
        with open(Files.file_rw, "w", encoding='utf-8') as myfile:
            myfile.write("datetime,touch_ball_reward,ball_to_goal_reward,player_to_ball_reward,ball_y_coordinate_reward,liu_distance_player_to_ball_reward,face_ball_reward,align_ball_to_goal,velocity_reward,save_boost_reward,combined_reward,penalized_reward\n")      
            
    action_maker = RLDiscreteAction()
    
    total_seconds = (core_timesteps * 0.0046875) * 14.171875
    
    penalized_reward  = PenalizedReward(
        touch_ball_reward=TouchBallReward(aerial_weight=0.2),
        ball_to_goal_reward=VelocityBallToGoalReward(),
        player_to_ball_reward=VelocityPlayerToBallReward(),
        ball_y_coordinate_reward=BallYCoordinateReward(),
        liu_distance_player_to_ball_reward=LiuDistancePlayerToBallReward(),
        face_ball_reward=FaceBallReward(),
        align_ball_to_goal=AlignBallGoal(),
        velocity_reward=VelocityReward(),
        save_boost_reward=SaveBoostReward(),
        
        touch_ball_weight=5.0,
        align_ball_to_goal_weight=3.0,
        ball_to_goal_weight=2.0,
		velocity_weight=1.5,
		liu_distance_player_to_ball_weight=1.4,
        player_to_ball_weight=1.2,
		face_ball_weight=1.1,
		save_boost_weight=1.0,
        ball_y_coordinate_weight=1.0)

    #game_speed=1
    logit("Loading Environment...")
    gym_env = rlgym.make(
                         terminal_conditions=[CustomTerminalCondition()],
                         reward_fn=penalized_reward,
                         obs_builder=CustomObsBuilder(),
                         action_parser=action_maker)
    
    # logit("loading Instance...") 
    # env_sb3 = SB3SingleInstanceEnv(gym_env)
    
    if os.path.exists(Files.file_model):
        logit("loading Model...")
        model = PPO.load(Files.file_model, env=gym_env)
        
    else:
        logit("creating model...") 
        policy_kwargs = dict(net_arch=[128, 128, 128])
        model = PPO("MlpPolicy", env=gym_env, learning_rate=0.0005, clip_range=0.2, verbose=0, batch_size=128, n_epochs=20, policy_kwargs=policy_kwargs)
        #0.001, 0.0005, 0.0001, or 0.00005

    task_start = thetime() 
    
    if model and gym_env:

        counter = 0
        incrementer = 40960
        onoff = 0
        interval_choice = 0
        
        while counter < 99999:

            if onoff == 0: interval_choice = 40_960
                
            if onoff == 1:
                interval_choice = 81_920
                onoff = 2

            prtit("learning <-",interval_choice)
            model.learn(interval_choice)
            
            counter = counter + 1
            core_timesteps = core_timesteps + incrementer
            total_seconds = (core_timesteps * 0.0046875) * 14.171875

            prtit("saving model...") 
            model.save(Files.file_model) 
            
            if onoff == 0: onoff = 1
            if onoff == 2: onoff = 0
            
            with open(Files.file_ts, "w") as myfile:
                myfile.write("ts="+str(core_timesteps))            

        try:
            gym_env.close()
        except:
            pass
        
        try:
            terminate('RocketLeague.exe')
        except:
            pass