import numpy as np

from rlgym.utils.gamestates import PlayerData, GameState

from rlgym.utils.obs_builders import ObsBuilder

from hwrlai.helpers.functions import get_seconds
from hwrlai.helpers.functions import save_gamestate
from hwrlai.helpers.functions import save_playerstate

from hwrlai.info.data import Field

from hwrlai.util.vec import Vec3

class CustomObsBuilder(ObsBuilder):

    def __init__(self, tick_skip=8, field_info=None):
        super().__init__()

        self.boost_timers = None
        self.tick_skip = tick_skip
        self.count_states_saved = 0

        self.future_ball_pos = None
        
        self.touchedlast = 0
        self.touchedamt = 0
        
        if field_info is None:
            self._boost_locations = np.array(Field.loc_boosts)
            self._boost_types = self._boost_locations[:, 2] > 72
        else:
            self._boost_locations = np.array([[bp.location.x, bp.location.y, bp.location.z]
                                              for bp in field_info.boost_pads[:field_info.num_boosts]])
            self._boost_types = np.array([bp.is_full_boost for bp in field_info.boost_pads[:field_info.num_boosts]])        

    def reset(self, initial_state: GameState): 
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player.ball_touched and get_seconds() > self.touchedlast:
            self.touchedlast = get_seconds() + 5
            self.touchedamt = self.touchedamt + 1
            
        if self.count_states_saved < 2500:
            save_playerstate(player,self.count_states_saved)
            save_gamestate(state,self.count_states_saved)
            self.count_states_saved = self.count_states_saved + 1     
            
        num_players = len(state.players)
        
        dAT_ball = state.ball.serialize()  
        
        ball_location = state.ball.position
        car_location = state.players[0].car_data.position
        boost_dists = np.linalg.norm(self._boost_locations - car_location, axis=1)
        
        if num_players == 2:    
            opp_location = state.players[1].car_data.position
        else:
            opp_location = np.array([0.0,0.0,0.0])

        ball_posV3 = Vec3(ball_location.tolist()[0],ball_location.tolist()[1],ball_location.tolist()[2],)
        car_posV3 = Vec3(car_location.tolist()[0],car_location.tolist()[1],car_location.tolist()[2],)
        opp_posV3 = Vec3(opp_location.tolist()[0],opp_location.tolist()[1],opp_location.tolist()[2],)
        
        goal_p_posV3 = Vec3(Field.POS_MY_GOAL)
        goal_o_posV3 = Vec3(Field.POS_EN_GOAL)

        dist_opp_car = car_posV3.dist(opp_posV3)
        
        dist_opp_ball = opp_posV3.dist(ball_posV3)
        dist_opp_opp_goal = opp_posV3.dist(goal_o_posV3)
        dist_opp_car_goal = opp_posV3.dist(goal_p_posV3)
        
        dist_car_ball = car_posV3.dist(ball_posV3)  
        dist_car_opp_goal = car_posV3.dist(goal_o_posV3)   
        dist_car_car_goal = car_posV3.dist(goal_p_posV3) 
        
        
        last_touch = state.last_touch

        if num_players == 2:    
            opp = state.players[1]
            dAT_opp = state.players[1].car_data.serialize()
            
            dAT_opp_ex1 = [round(opp.boost_amount*100),
                              opp.boost_pickups,
                              opp.match_goals,
                              opp.match_shots]     
            
            dAT_opp_ex2 = [int(opp.ball_touched),
                              int(opp.has_flip),
                              int(opp.has_jump),
                              int(opp.is_demoed),
                              int(opp.on_ground)]    
        else:
            dAT_opp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            dAT_opp_ex1 = [0,0,0,0]     
            dAT_opp_ex2 = [0,0,0,0,0] 

        dAT_player = player.car_data.serialize()
        

        dAT_player_ex1 = [round(player.boost_amount*100),
                          player.boost_pickups,
                          player.match_goals,
                          player.match_shots] 

        dAT_player_ex2 = [int(player.ball_touched),
                          int(player.has_flip),
                          int(player.has_jump),
                          int(player.is_demoed),
                          int(player.on_ground),
                          int(last_touch)]

        dAT_distances = [dist_opp_car,
                         dist_opp_ball,
                         dist_opp_opp_goal,
                         dist_opp_car_goal,
                         dist_car_ball,
                         dist_car_opp_goal,
                         dist_car_car_goal]        
        
        obs = []
        obs += dAT_ball        
        obs += dAT_player
        obs += list(dAT_player_ex1)
        obs += list(dAT_player_ex2)
        obs += list(boost_dists)
        obs += list(map(lambda x: 1 if x else 0, self._boost_types))
        obs += list(previous_action)
        obs += list(dAT_distances)
        obs += dAT_opp
        obs += list(dAT_opp_ex1)
        obs += list(dAT_opp_ex2)           

        return np.asarray(obs, dtype=np.float32)