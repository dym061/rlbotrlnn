import time


from rlbot.utils.structures import game_interface
from rlgym.utils.gamestates import GameState

from rlgym.utils.terminal_conditions import TerminalCondition

from hwrlai.helpers.functions import is_shot_on_goal
from hwrlai.helpers.functions import logit

from hwrlai.info.data import Field

class CustomTerminalCondition(TerminalCondition):
    
    def __init__(self):
        self.start_time = None
        self.prev_pos_x = None
        self.prev_pos_y = None
        self.maxspeed = 0
        
        self.count_touched = 0
        
        self.goals_scored = 0

    def reset(self, initial_state: GameState):
        self.start_time = time.time()
        self.prev_pos_x = initial_state.players[0].car_data.position[0]
        self.prev_pos_y = initial_state.players[0].car_data.position[1]

    def is_terminal(self, current_state: GameState) -> bool:
        
        scored = 0
        
        goals_scored = current_state.players[0].match_goals
        if goals_scored > self.goals_scored:
            self.goals_scored = goals_scored
            scored = 1
        
        current_time = time.time()
        
        ball_pos_x = current_state.ball.position[0]
        ball_pos_y = current_state.ball.position[1]
        ball_pos_z = current_state.ball.position[2]
        
        if(is_shot_on_goal(current_state,Field)):
            logit("Shot On Goal!")
        
        #stop this driving in circles nonsense
        if current_time - self.start_time >= 3:
            
            player_data = current_state.players[0]

            curr_pos_x = player_data.car_data.position[0]
            curr_pos_y = player_data.car_data.position[1]
            
            distance_moved_x = (curr_pos_x - self.prev_pos_x)
            distance_moved_y = (curr_pos_y - self.prev_pos_y)
            
            if distance_moved_x < 200 and distance_moved_y < 200:
                return True
            
            self.prev_pos_x = curr_pos_x
            self.prev_pos_y = curr_pos_y
            
            self.start_time = current_time
            
        #reset if the ball is in the goal
        if scored:
            logit("Goal!"+"["+str(self.goals_scored)+"]")
            scored = 0
            return True
        
        if ball_pos_y <= int(Field.POS_MY_GOAL[1]) and (ball_pos_x >= -850 and ball_pos_x < 850) and (ball_pos_z >= 15 and ball_pos_z <= Field.GOAL_HEIGHT):
            logit("Own Goal!")
            return True        
        
        # #reset if touched 5 times
        # if current_state.last_touch != -1:
        #     self.count_touched = self.count_touched + 1
            
        #     if self.count_touched == 5:
        #         self.count_touched = 0
        #         return True 