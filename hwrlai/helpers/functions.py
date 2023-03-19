import math
import numpy as np
import os
import pickle
import random
import time

from datetime import datetime
from hwrlai.util.vec import Vec3

from hwrlai.info.data import Field, Files

def get_filesize(filein):
    return os.path.getsize(filein)

def make_state_dict(state):

    data_player = state.players[0]
    data_ball = state.ball
    
    score_player = state.blue_score
    score_opp = state.orange_score
    
    kill_death = 0

    if score_player > score_opp:
        kill_death = 1
    else:
        if score_player == score_opp:
            kill_death = 0
        else:        
            if score_player < score_opp:
                kill_death = -1
            else:
                kill_death = 0

    ball_pos = Vec3(data_ball.position[0],data_ball.position[1],data_ball.position[2])
    car_pos = Vec3(data_player.car_data.position[0],data_player.car_data.position[1],data_player.car_data.position[2])
    boost_amount = data_player.boost_amount
    
    if len(state.players) > 1: data_opp = state.players[1]            
    if len(state.players) > 1: opp_pos = Vec3(data_opp.car_data.position[0],data_opp.car_data.position[1],data_opp.car_data.position[2])
    if len(state.players) > 1: distance_to_opp = car_pos.dist(opp_pos)             

    #angle_to_ball = calc_angl_to(data_player.car_data.position,data_ball.position)
    #angle_to_goal = calc_angl_to(Field.POS_EN_GOAL,data_player.car_data.position)

    distance_to_ball = car_pos.dist(ball_pos)
    
    distance_to_goal = ball_pos.dist(Field.POS_EN_GOAL)     
    
    forward_velocity = np.dot(state.players[0].car_data.linear_velocity, [-1, 0, 0])
  
    if len(state.players) > 1: 
        return \
            {
            "distance_to_ball" : int(distance_to_ball),
            "distance_to_goal" : int(distance_to_goal),
            "boost_amount" : round(boost_amount,2),
            "kill_death" : int(kill_death),
            "forward_velocity" : int(forward_velocity),
            "distance_to_opp" : int(distance_to_opp)
            }
            
    if len(state.players) == 1: 
        return \
            {
            "distance_to_ball" : int(distance_to_ball),
            "distance_to_goal" : int(distance_to_goal),
            "boost_amount" : round(boost_amount,2),
            "kill_death" : int(kill_death),
            "forward_velocity" : int(forward_velocity),
            }    

def calc_angl_to(player_loc, ball_loc):
    try:
        player_pos = player_loc
        ball_pos = ball_loc
        delta_x = ball_pos[0] - player_pos[0]
        delta_y = ball_pos[1] - player_pos[1]
        angle = math.atan2(delta_y, delta_x) * 180 / math.pi
        if angle < 0:
            angle += 360
        return angle
    
    except:
        return 0 
    
def remove_spa_from_npa(npa):
    
    temp = npa
    try:
        temp = np.squeeze(temp)
        temp = temp[~np.all(temp == 0, axis=0)]
        temp = str(temp)
        temp = temp.replace("  "," ").replace("[[ ","[[")
        temp = np.array(temp)
        
        return temp
        
    except:
        
        return temp    

def get_seconds():
    return time.time()

def act_reshape(actions,bins):

    if type(actions) == list:
        actions = np.array(actions)

    actions = actions.reshape((-1, 8))
    actions[..., :5] = actions[..., :5] / (bins // 2) - 1
    
    return actions

def simulate_game_actions():

    base_action = [0,0,0,0,0,0,0,0]
    
    base_action[0] = random.randint(1, 2) #1
    base_action[1] = random.randint(1, 2) #2
    base_action[2] = random.randint(1, 2) #3
    base_action[3] = random.randint(1, 2) #4
    base_action[4] = random.randint(1, 2) #5
    base_action[5] = random.randint(0, 1) #6
    base_action[6] = random.randint(0, 1) #7
    base_action[7] = random.randint(0, 1) #8
    
    return base_action

def create_random_actions():

    base_action = [0,0,0,0,0,0,0,0]
    
    base_action[0] = random.randint(-1, 1) #1
    base_action[1] = random.randint(-1, 1) #2
    base_action[2] = random.randint(-1, 1) #3
    base_action[3] = random.randint(-1, 1) #4
    base_action[4] = random.randint(-1, 1) #5
    base_action[5] = random.randint(0, 1) #6
    base_action[6] = random.randint(0, 1) #7
    base_action[7] = random.randint(0, 1) #8
    
    return base_action

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def dR(stuff):
    print("===")
    print("type:::",type(stuff))
    print("value::: ",namestr(stuff,globals()), ":", stuff)
    print("===")
    print("dir::: ",namestr(stuff,globals()), ":", dir(stuff))
    print("===")
    print("")

def save_gamestate(obj,inc):
    with open(Files.file_gs+"_"+str(inc), 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def save_playerstate(obj,inc):
    with open(Files.file_ps+"_"+str(inc), 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)        

def get_player_speed(value):
    try:
        return np.linalg.norm(value)
    except:
        return 0
 
def is_shot_on_goal(state,field) -> bool:
    
    ball_pos = state.ball.position
    
    ball_pos_x = ball_pos[0]     
    ball_pos_y = ball_pos[1]  
    ball_pos_z = ball_pos[2]
    
    ball_vel = state.ball.linear_velocity[2] 
    goal_width = field.GOAL_WIDTH + field.BALL_RADIUS

    if ball_vel > 0:
        # Ball is moving towards opponent's goal
        if abs(ball_pos_x) < goal_width/2 and ball_pos_z > (field.GOAL_HEIGHT - field.BALL_RADIUS):
            # Ball is within the goalposts and has crossed the goal line
            if abs(ball_pos_y) > Field.POS_EN_GOAL[1]:
                # Ball is between the goalposts
                return True

    return False  

def convert_seconds_to_duration(seconds):
    time = seconds
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return ("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))

def con_str_2_n_or_f(string):
    
    if str(type(string)) == "<class 'int'>":
        return string
    
    else:
        
        if str(type(string)) == "<class 'float'>":
            return string
        
        else:
    
            try:
                return int(string)
            except:
                
                try:
                    return float(string)
                except:
                    return string

def thetime():
    return int(time.time())
   
def logit(data):
    now = str(datetime.now())

    with open(Files.file_log, "a") as myfile:
        print(now+" "+str(data))
        myfile.write(now+" "+str(data)+"\n")  
        
def prtit(string,data=""):
    now = str(datetime.now())
    print(now,string,data)

def terminate(ProcessName):
    os.system('taskkill /IM "' + ProcessName + '" /F')