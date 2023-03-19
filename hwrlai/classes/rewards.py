import os

from datetime import datetime


from rlgym.utils.gamestates import  GameState

from rlgym.utils.reward_functions import RewardFunction

from hwrlai.helpers.functions import prtit

from hwrlai.info.data import Files

class NormalReward(RewardFunction):
    def __init__(self, 
                 touch_ball_reward,
                 ball_to_goal_reward,
                 player_to_ball_reward,
                 ball_y_coordinate_reward,
                 liu_distance_player_to_ball_reward,
                 face_ball_reward,
                 align_ball_to_goal,
                 velocity_reward,
                 save_boost_reward,
                 
                 touch_ball_weight,
                 ball_to_goal_weight,
                 player_to_ball_weight,
                 ball_y_coordinate_weight,
                 liu_distance_player_to_ball_weight,
                 face_ball_weight,
                 align_ball_to_goal_weight,
                 velocity_weight,
                 save_boost_weight):
        
        self.touch_ball_reward=touch_ball_reward
        self.ball_to_goal_reward=ball_to_goal_reward
        self.player_to_ball_reward=player_to_ball_reward
        self.ball_y_coordinate_reward=ball_y_coordinate_reward
        self.liu_distance_player_to_ball_reward=liu_distance_player_to_ball_reward
        self.face_ball_reward=face_ball_reward
        self.align_ball_to_goal=align_ball_to_goal
        self.velocity_reward=velocity_reward
        self.save_boost_reward=save_boost_reward

        self.touch_ball_weight=touch_ball_weight
        self.ball_to_goal_weight=ball_to_goal_weight
        self.player_to_ball_weight=player_to_ball_weight
        self.ball_y_coordinate_weight=ball_y_coordinate_weight
        self.liu_distance_player_to_ball_weight=liu_distance_player_to_ball_weight
        self.face_ball_weight=face_ball_weight
        self.align_ball_to_goal_weight=align_ball_to_goal_weight
        self.velocity_weight=velocity_weight
        self.save_boost_weight=save_boost_weight

        self.max_combined_reward = 0
        self.max_penalized_reward = 0
        
        self.loaded_prev_rewards = 0

    def get_reward(self, game_state: GameState, action: int, new_game_state: GameState) -> float:
        
        touch_ball_reward                   = self.touch_ball_weight                    *self.touch_ball_reward.get_reward(game_state, action, new_game_state)
        ball_to_goal_reward                 = self.ball_to_goal_weight                  *self.ball_to_goal_reward.get_reward(game_state, action, new_game_state)
        player_to_ball_reward               = self.player_to_ball_weight                *self.player_to_ball_reward.get_reward(game_state, action, new_game_state)
        ball_y_coordinate_reward            = self.ball_y_coordinate_weight             *self.ball_y_coordinate_reward.get_reward(game_state, action, new_game_state)
        liu_distance_player_to_ball_reward  = self.liu_distance_player_to_ball_weight   *self.liu_distance_player_to_ball_reward.get_reward(game_state, action, new_game_state)
        face_ball_reward                    = self.face_ball_weight                     *self.face_ball_reward.get_reward(game_state, action, new_game_state)
        align_ball_to_goal                  = self.align_ball_to_goal_weight            *self.align_ball_to_goal.get_reward(game_state, action, new_game_state)
        velocity_reward                     = self.velocity_weight                      *self.velocity_reward.get_reward(game_state, action, new_game_state)
        save_boost_reward                   = self.save_boost_weight                    *self.save_boost_reward.get_reward(game_state, action, new_game_state)

        combined_reward=touch_ball_reward+\
                        ball_to_goal_reward+\
                        player_to_ball_reward+\
                        ball_y_coordinate_reward+\
                        liu_distance_player_to_ball_reward+\
                        face_ball_reward+\
                        align_ball_to_goal+\
                        velocity_reward+\
                        save_boost_reward

        return combined_reward
    
    def reset(self, initial_state: GameState):
        pass
        
class PenalizedReward(NormalReward):
        
    def get_reward(self, game_state: GameState, action: int, new_game_state: GameState) -> float:

        # Get the individual reward values from the superclass
        touch_ball_reward = self.touch_ball_weight * self.touch_ball_reward.get_reward(game_state, action, new_game_state)
        ball_to_goal_reward = self.ball_to_goal_weight * self.ball_to_goal_reward.get_reward(game_state, action, new_game_state)
        player_to_ball_reward = self.player_to_ball_weight * self.player_to_ball_reward.get_reward(game_state, action, new_game_state)
        ball_y_coordinate_reward = self.ball_y_coordinate_weight * self.ball_y_coordinate_reward.get_reward(game_state, action, new_game_state)
        liu_distance_player_to_ball_reward = self.liu_distance_player_to_ball_weight * self.liu_distance_player_to_ball_reward.get_reward(game_state, action, new_game_state)
        face_ball_reward = self.face_ball_weight * self.face_ball_reward.get_reward(game_state, action, new_game_state)
        align_ball_to_goal = self.align_ball_to_goal_weight * self.align_ball_to_goal.get_reward(game_state, action, new_game_state)
        velocity_reward = self.velocity_weight * self.velocity_reward.get_reward(game_state, action, new_game_state)
        save_boost_reward = self.save_boost_weight * self.save_boost_reward.get_reward(game_state, action, new_game_state)

        # Combine the individual reward values
        combined_reward =   touch_ball_reward +\
                            ball_to_goal_reward +\
                            player_to_ball_reward +\
                            ball_y_coordinate_reward +\
                            liu_distance_player_to_ball_reward +\
                            face_ball_reward + align_ball_to_goal +\
                            velocity_reward +\
                            save_boost_reward
                            
        # Calculate the average of all the reward weights
        reward_weights = [self.touch_ball_weight, 
                          self.ball_to_goal_weight, 
                          self.player_to_ball_weight, 
                          self.ball_y_coordinate_weight, 
                          self.liu_distance_player_to_ball_weight, 
                          self.face_ball_weight, 
                          self.align_ball_to_goal_weight, 
                          self.velocity_weight,
                          self.save_boost_weight]
        
        average_weight = sum(reward_weights) / len(reward_weights)

        # Penalize the combined reward based on the average weight and a constant penalty factor of 0.01
        penalty = average_weight * 0.25
        
        penalized_reward = combined_reward - penalty

        if self.loaded_prev_rewards == 0:

            if os.path.exists(Files.file_cr):            
                with open(Files.file_cr) as j:
                    content = j.readlines()
                    self.max_combined_reward = float(content[0].split("=")[1])
                    prtit("loaded previous cr")
                
            if os.path.exists(Files.file_pr):            
                with open(Files.file_pr) as j:
                    content = j.readlines()
                    self.max_penalized_reward = float(content[0].split("=")[1])     
                    prtit("loaded previous pr")
                    
            self.loaded_prev_rewards = 1

        if combined_reward > self.max_combined_reward:
            now = str(datetime.now())
            self.max_combined_reward = combined_reward
            with open(Files.file_cr, "w") as myfile:
                myfile.write("cr="+str(self.max_combined_reward))   
            
        if penalized_reward > self.max_penalized_reward:
            now = str(datetime.now())
            self.max_penalized_reward = penalized_reward
            with open(Files.file_pr, "w") as myfile:
                myfile.write("pr="+str(self.max_penalized_reward)) 

        if 1==1:
            
            if touch_ball_reward > 0:
                
                now = str(datetime.now())

                if os.path.exists(Files.file_rw): 
                    with open(Files.file_rw, "a", encoding='utf-8') as myfile:
                        myfile.write(now+","+str(touch_ball_reward)+","+str(ball_to_goal_reward)+","+str(player_to_ball_reward)+","+str(ball_y_coordinate_reward)+","+str(liu_distance_player_to_ball_reward)+","+str(face_ball_reward)+","+str(align_ball_to_goal)+","+str(velocity_reward)+","+str(save_boost_reward)+","+str(combined_reward)+","+str(penalized_reward)+"\n")      
 
        return penalized_reward