import pandas as pd
import math
import nashpy as nash
import numpy as np
import random
import warnings
warnings.simplefilter("ignore")

T = 0.1
#################################################
# --------------------TODO----------------------#
#################################################
DELTA = 1.0
MAX_DELTA_YAW = 2.0
SAFE_DISTANCE = 15
MAX_V_ACC = 40.0
MAX_V_VEL = 50.0

COOPERATE_COEF = 1
AGGRESIVE_COEF = 1


output_path = 'trajectory/safe_15.csv'
# METHOD = 'rule_based' # 'rule_based' or 'game_theory'
METHOD = 'game_theory'
#################################################

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class Vehicle:
    def __init__(self, predefined_trajectory, _id, env):
        # state
        self.PREDFINED = -1
        self.FOLLOW = 0
        self.LANE_CHANGE = 1 # the target vehicle changing lane (vehicle A)
        self.LANE_CHANGE_LAG = 2 # the lag vehicle which the target vehicle is going to insert in front of (vehicle B)
        self.LANE_KEEP = 3

        self.LANE_R = 0
        self.LANE_L = 1
        
        self.predefined_trajectory = predefined_trajectory.set_index('Frame_ID') # Dataframe: (Frame_ID,Local_X,Local_Y,v_Vel)
        self.num_predifined_frame = len(predefined_trajectory)
        self.id = _id
        self.frame = 0
        
        self.loc_x = self.predefined_trajectory.iloc[0]['Local_X']
        self.loc_y = self.predefined_trajectory.iloc[0]['Local_Y']
        self.v_vel = self.predefined_trajectory.iloc[0]['v_Vel']
        self.v_acc = 0
        self.yaw = 0

        self.next_loc_x = None
        self.next_loc_y = None
        self.next_v_vel = None
        self.next_v_acc = None
        self.next_yaw = None
        self.next_state = None
        
        self.trajectory = [[self.frame, self.loc_x, self.loc_y, self.v_vel]] # list of [frame, x, y, v]
        self.state = self.PREDFINED if self.num_predifined_frame > 1 else self.LANE_KEEP
        self.lane_id = self.LANE_R if abs(self.loc_x - env.lanes[0].middle) < abs(self.loc_x - env.lanes[1].middle) else self.LANE_L
        self.target_lane_id = None
        self.lead_vehicle = None
        self.side_lead_vehicle = None
        self.side_lag_vehicle = None

    def nextStep(self, env):
        if self.state == self.PREDFINED:
            next_loc_x = self.predefined_trajectory.iloc[self.frame+1]['Local_X']
            next_loc_y = self.predefined_trajectory.iloc[self.frame+1]['Local_Y']
            next_v_vel = self.predefined_trajectory.iloc[self.frame+1]['v_Vel']
            next_yaw = math.atan((next_loc_x - self.loc_x) / (next_loc_y - self.loc_y)) * 180 / math.pi
            next_v_acc = (next_v_vel - self.v_vel) / T
            
            # check collision
            if self.lead_vehicle != None:
                lead_distance = self.cal_distance(self.lead_vehicle)
            else:
                lead_distance = float('inf')
            
            # if next_yaw - self.yaw < -1.7 and self.id == 1103: # consider only vehicle A (predifined traj)
                # self.state = self.LANE_CHANGE
                # self.target_lane_id = 1 if self.lane_id == 0 else 0

            if self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE and METHOD == 'game_theory':
                self.next_state = self.LANE_CHANGE_LAG
            elif self.frame >= self.num_predifined_frame - 2:
                self.next_state = self.LANE_KEEP
            elif lead_distance <= SAFE_DISTANCE:
                self.next_state = self.FOLLOW
                next_v_acc = -MAX_V_ACC
            else:
                self.next_state = self.PREDFINED
        
        elif self.state == self.FOLLOW:

            # present lead vehicle distance
            if self.lead_vehicle != None:
                # lead_distance = self.cal_distance(next_loc_x, next_loc_y, self.lead_vehicle)
                lead_distance = self.cal_distance(self.lead_vehicle)
            else:
                lead_distance = float('inf')
            
            next_yaw = 0.0
            if lead_distance <= SAFE_DISTANCE:
                next_v_acc = -min(MAX_V_ACC, self.v_vel / T)
            elif lead_distance >= 3 * SAFE_DISTANCE:
                next_v_acc = min(max(self.v_acc, 0) + 1.0, MAX_V_ACC, (MAX_V_VEL - self.v_vel) / T)
            else:
                next_v_acc = 0.0
            
            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel(next_v_acc, next_yaw)

            if self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE and METHOD == 'game_theory':
                self.next_state = self.LANE_CHANGE_LAG
            else:
                self.next_state = self.FOLLOW

        elif self.state == self.LANE_CHANGE:
            
            # METHOD 1: rule-based
            if METHOD == 'rule_based':
                next_v_acc, next_yaw = self.rule_based_lane_change()
            # METHOD 2: game theory
            elif METHOD == 'game_theory':
                print('*' * 150 + '\n' + '{} starting game'.format(self.id))
                next_v_acc, next_yaw, lag_next_v_acc, lag_next_yaw = self.game_theory_lane_change('target',self, self.side_lag_vehicle, self.side_lead_vehicle)
                print('*' * 150)
            
            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel(next_v_acc, next_yaw)
            # check collision
            if self.side_lag_vehicle != None and self.side_lag_vehicle.side_lead_vehicle == self:
                if METHOD == 'game_theory':
                    side_lag_distance = self.cal_distance_next(next_loc_x, next_loc_y, self.side_lag_vehicle, lag_next_v_acc, lag_next_yaw)
                elif METHOD == 'rule_based':
                    side_lag_distance = self.cal_distance(self.side_lag_vehicle)
                # side_lag_distance = self.cal_distance(self.loc_x, self.loc_y, self.side_lag_vehicle, timestep='present')
            else:
                side_lag_distance = float('inf')
            if self.side_lead_vehicle != None:
                side_lead_distance = self.cal_distance(self.side_lead_vehicle)
                # side_lead_distance = self.cal_distance(self.loc_x, self.loc_y, self.side_lead_vehicle, timestep='present')
            else:
                side_lead_distance = float('inf')
            
            print('A check side lag distance:', side_lag_distance)
            if side_lag_distance <= SAFE_DISTANCE or side_lead_distance <= SAFE_DISTANCE:
                print(side_lag_distance, side_lead_distance)
                print('\n' + '*' * 10 + 'CANCEL LANE CHANGE' + '*' * 10 + '\n')
                self.next_state = self.LANE_KEEP
                self.target_lane_id = None
            elif self.target_lane_id == 1 and next_loc_x <= env.lanes[1]._in:
                self.next_state = self.LANE_KEEP
                self.lane_id = self.LANE_L
                self.target_lane_id = None
            elif self.target_lane_id == 0 and next_loc_x >= env.lanes[0]._in:
                self.next_state = self.LANE_KEEP
                self.lane_id = self.LANE_R
                self.target_lane_id = None
            elif self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE and METHOD == 'game_theory':
                self.next_state = self.LANE_CHANGE_LAG
            else:
                self.next_state = self.LANE_CHANGE

        elif self.state == self.LANE_KEEP:
            next_yaw = 2.0 * (env.lanes[self.lane_id].middle - self.loc_x)
            if abs(next_yaw - self.yaw) > MAX_DELTA_YAW:
                if next_yaw > self.yaw:
                    next_yaw = self.yaw + MAX_DELTA_YAW
                else:
                    next_yaw = self.yaw - MAX_DELTA_YAW
            next_v_acc = 0.0
            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel(next_v_acc, next_yaw)
            # check collision
            if self.lead_vehicle != None:
                lead_distance = self.cal_distance(self.lead_vehicle)
            else:
                lead_distance = float('inf')
            if lead_distance <= SAFE_DISTANCE:
                next_v_acc = -min(MAX_V_ACC, next_v_vel / T)
            else:
                next_v_acc = 0.0
            if self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE and METHOD == 'game_theory':
                self.next_state = self.LANE_CHANGE_LAG
            elif next_loc_x <= env.lanes[self.lane_id].middle + DELTA and next_loc_x >= env.lanes[self.lane_id].middle - DELTA:
                self.next_state = self.FOLLOW
            else:
                self.next_state = self.LANE_KEEP
        elif self.state == self.LANE_CHANGE_LAG:
            print('*' * 150 + '\n' + '{} starting game'.format(self.id))
            _, _, next_v_acc, next_yaw = self.game_theory_lane_change('lag', self.side_lead_vehicle, self.side_lead_vehicle.side_lag_vehicle, self.side_lead_vehicle.side_lead_vehicle)
            print('*' * 150)
            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel(next_v_acc, next_yaw)
            if self.side_lead_vehicle ==  None or self.side_lead_vehicle.state != self.LANE_CHANGE:
                self.next_state = self.FOLLOW
            else:
                self.next_state = self.LANE_CHANGE_LAG

        self.next_loc_x = next_loc_x
        self.next_loc_y = next_loc_y
        self.next_v_vel = next_v_vel
        self.next_v_acc = next_v_acc
        self.next_yaw = next_yaw

        self.frame += 1
        self.trajectory.append([self.frame, self.next_loc_x, self.next_loc_y, self.next_v_vel])
    def update(self):
        self.loc_x = self.next_loc_x
        self.loc_y = self.next_loc_y
        self.v_vel = self.next_v_vel
        self.v_acc = self.next_v_acc
        self.yaw = self.next_yaw
        self.state = self.next_state

    def cal_loc_vel(self, next_v_acc, next_yaw):
        # next_loc_x = self.loc_x +  math.sin(self.yaw / 180 * math.pi) * (self.v_vel * T + 0.5 * self.v_acc * T**2 )
        # next_loc_y = self.loc_y +  math.cos(self.yaw / 180 * math.pi) * (self.v_vel * T + 0.5 * self.v_acc * T**2 )
        # next_v_vel = self.v_vel + self.v_acc * T
        next_loc_x = self.loc_x +  math.sin(next_yaw / 180 * math.pi) * (self.v_vel * T + 0.5 * next_v_acc * T**2 )
        next_loc_y = self.loc_y +  math.cos(next_yaw / 180 * math.pi) * (self.v_vel * T + 0.5 * next_v_acc * T**2 )
        next_v_vel = self.v_vel + next_v_acc * T

        return next_loc_x, next_loc_y, next_v_vel

    def cal_distance(self, ref_vehicle): # timestep 'present' or 'now'
        # if timestep == 'next':
        #     pred_ref_loc_x = ref_vehicle.loc_x + math.sin(ref_vehicle.yaw / 180 * math.pi) * (ref_vehicle.v_vel * T + 0.5 * ref_vehicle.v_acc * T**2 )
        #     pred_ref_loc_y = ref_vehicle.loc_y + math.cos(ref_vehicle.yaw / 180 * math.pi) * (ref_vehicle.v_vel * T + 0.5 * ref_vehicle.v_acc * T**2 )
        #     return ((next_loc_x - pred_ref_loc_x) ** 2 + (next_loc_y - pred_ref_loc_y) ** 2) ** 0.5
        # else:
        return ((self.loc_x - ref_vehicle.loc_x) ** 2 + (self.loc_y - ref_vehicle.loc_y) ** 2) ** 0.5
    def cal_distance_next(self, next_loc_x, next_loc_y, ref_vehicle, ref_next_v_acc, ref_next_yaw):
        pred_ref_v_vel = ref_vehicle.v_vel + ref_next_v_acc * T
        pred_ref_loc_x = ref_vehicle.loc_x + math.sin(ref_next_yaw / 180 * math.pi) * (pred_ref_v_vel * T + 0.5 * ref_next_v_acc * T**2 )
        pred_ref_loc_y = ref_vehicle.loc_y + math.cos(ref_next_yaw / 180 * math.pi) * (pred_ref_v_vel * T + 0.5 * ref_next_v_acc * T**2 )
        print(f'A predict B next loc_x: {pred_ref_loc_x}, loc_y: {pred_ref_loc_y}, v_acc: {ref_next_v_acc}, yaw: {ref_next_yaw}')
        return ((pred_ref_loc_x - next_loc_x) ** 2 + (pred_ref_loc_y - next_loc_y) ** 2) ** 0.5
    def rule_based_lane_change(self):
        _sign = -1 if self.target_lane_id == 1 else 1
        next_yaw = self.yaw + _sign * MAX_DELTA_YAW
        if self.side_lag_vehicle != None:
            Acc_target_t = 2 / T ** 2 / math.cos(self.yaw * 180 / math.pi)* (SAFE_DISTANCE \
            + self.side_lag_vehicle.loc_y + self.side_lag_vehicle.v_vel * T + 0.5 * self.side_lag_vehicle.v_acc *  T**2 \
            - self.loc_y - math.cos(self.yaw * 180 / math.pi) * self.v_vel * T)
            Acc_target_t = max(Acc_target_t, 0) # v_acc_target > Acc_target_t is feasible
        else:
            Acc_target_t = 0.0
        next_v_acc = min(MAX_V_ACC, Acc_target_t)

        return next_v_acc, next_yaw

    def game_theory_lane_change(self, position, target_vehicle, side_lag_vehicle, side_lead_vehicle): # position: 'target' or 'lag'
        # Reference: Modeling Lane-Changing Behavior in a Connected Environment: A Game Theory Approach
        _sign = -1 if target_vehicle.target_lane_id == 1 else 1
        # If no side lag vehicle, target vehicle just change the lane, doesn't need to play game
        if side_lag_vehicle == None or side_lag_vehicle.side_lead_vehicle != target_vehicle:
            new_v_acc = 0.0
            new_yaw = self.yaw + _sign * MAX_DELTA_YAW
            print('No Game, Target chose LANE CHANGE')
            return new_v_acc, new_yaw, None, None

        # If there is a side lag vehicle, play game
        # For target vehicle
        # Acc_target
        Acc_target_t = 2 / T ** 2 / math.cos(target_vehicle.yaw * 180 / math.pi)* ((SAFE_DISTANCE + 2.0) \
            - side_lag_vehicle.loc_y - side_lag_vehicle.v_vel * T - 0.5 * side_lag_vehicle.v_acc *  T**2 \
            + target_vehicle.loc_y + math.cos(target_vehicle.yaw * 180 / math.pi) * target_vehicle.v_vel * T)
        # Acc_target_t = max(Acc_target, 0) # v_acc_target > Acc_target_t is feasible
        
        # Acc_lead
        if side_lead_vehicle != None:
            Acc_lead_t = 2 / T ** 2 / math.cos(target_vehicle.yaw * 180 / math.pi)* (-(SAFE_DISTANCE + 2.0) \
             + side_lead_vehicle.loc_y + side_lead_vehicle.v_vel * T - 0.5 * side_lead_vehicle.v_acc *  T**2 \
             - target_vehicle.loc_y - math.cos(target_vehicle.yaw * 180 / math.pi) * target_vehicle.v_vel * T)
            # Acc_lead_t = -min(Acc_target, 0) # v_acc_target < Acc_lead_t is feasible
        if side_lead_vehicle != None and target_vehicle.lead_vehicle != None:
            delta_v_vel = target_vehicle.side_lead_vehicle.v_vel - target_vehicle.lead_vehicle.v_vel
        else:
            delta_v_vel = 0

        Acc_cf = 0.0

        # For lag vehicle
        Acc_target_l = 2 / T ** 2 * (-(SAFE_DISTANCE + 2.0) \
         + target_vehicle.loc_y + math.cos(target_vehicle.yaw * 180 / math.pi) * (target_vehicle.v_vel * T + 0.5 * target_vehicle.v_acc *  T**2) \
         - side_lag_vehicle.loc_y - side_lag_vehicle.v_vel * T)
        # Acc_target_l = -min(Acc_target_l, 0) # v_acc_lag < Acc_target_l is feasible

        if side_lead_vehicle != None and side_lag_vehicle != None:
            Acc_lead_l = 2 / T ** 2 * (-(SAFE_DISTANCE + 2.0) \
                + side_lead_vehicle.loc_y + side_lead_vehicle.v_vel * T + 0.5 * side_lead_vehicle.v_acc * T ** 2\
                - side_lag_vehicle.loc_y - side_lag_vehicle.v_vel * T)
            # Acc_lead_l = -min(Acc_lead_l, 0) # v_acc_lag < Acc_lead_l is feasible
        else:
            Acc_lead_l = 0

        Acc_target_t = max(Acc_target_t, 0)
        Acc_lead_t = min(Acc_lead_t, 0)
        Acc_target_l = min(Acc_target_l, 0)
        Acc_lead_l = min(Acc_lead_l, 0)

        Acc_target_Y = min(Acc_target_l, -MAX_V_ACC)
        Acc_lead_Y = min(Acc_lead_l, -MAX_V_ACC)
        
        #################################################
        # --------------------TODO----------------------#
        #################################################
        a110, a111, a112, a113, e11 = 4000*AGGRESIVE_COEF, -1, 0, 1, 0 # change, accelerate
        a210, a211, a212, a213, e21 = 5000*AGGRESIVE_COEF, -1, 0, 1, 0 # change, decelerate
        a120, a121, e12 = -5000*AGGRESIVE_COEF, 0, 1e-8                # not change, accelerate
        a220, a221, e22 = -6000*AGGRESIVE_COEF, 0, 1e-8                # not change, decelerate
        b110, b111, d11 = -4000*COOPERATE_COEF, -1, 0                  # change, accelerate
        b120, b121, d12 = -2000*COOPERATE_COEF, -1, 0                      # not change, accelerate
        b210, b211, d21 = 6000*COOPERATE_COEF, 1, 0                    # change, decelerate
        b220, b221, d22 = 5000*COOPERATE_COEF, 1, 0                       # not change, decelerate
        #################################################

        print('\tacc target t:', Acc_target_t, Acc_target_t > 0)
        print('\tacc target l:', Acc_target_l, Acc_target_l < 0)
        print('\tacc lead l:', Acc_lead_l, Acc_lead_l < 0)
        print('\tacc target y:', Acc_target_Y, Acc_target_Y < 0)
        print('\tacc lead y:', Acc_lead_Y, Acc_lead_Y < 0)
        print()
        
        

        P11 = a110 + a111 * Acc_target_t  + a112 * Acc_lead_t + a113 * delta_v_vel + e11
        P21 = a210 + a211 * Acc_target_t + a212 * Acc_lead_t + a213 * delta_v_vel + e21
        P12 = a120 + a121 * Acc_cf + e12
        P22 = a220 + a221 * Acc_cf + e22
        Q11 = b110 + b111 * Acc_target_l + d11
        Q12 = b120 + b121 * Acc_lead_l + d12
        Q21 = b210 + b211 * Acc_target_Y + d21
        Q22 = b220 + b221 * Acc_lead_Y + d22
        print('target payload')
        print(f'\t{P11} {P12}\n\t{P21} {P22}')
        print('lag payload')
        print(f'\t{Q11} {Q12}\n\t{Q21} {Q22}')
        print()
        
        # game = nash.Game(np.array([[P11, P12], [P21, P22]]), np.array([[Q11, Q12], [Q21, Q22]]))
        # B: row player, A: column player
        game = nash.Game(np.array([[Q11, Q12], [Q21, Q22]]), np.array([[P11, P12], [P21, P22]]))
        # equilibria = game.vertex_enumeration()
        equilibria = game.support_enumeration()
        eqs = []
        for eq in equilibria:
            eqs.append(eq)
        
        # chosen_eq = random.choice(eqs)
        chosen_eq = eqs[0]
       
        target_strategy = chosen_eq[1]
        lag_strategy = chosen_eq[0]
        
        # target_action = np.random.choice(np.arange(2), p=softmax(target_strategy))
        # lag_action = np.random.choice(np.arange(2), p=softmax(lag_strategy))
        target_action = np.argmax(target_strategy)
        lag_action = np.argmax(lag_strategy)
        id2action_t = {0: 'LANE CHANGE', 1: 'NOT LANE CHANGE'}
        id2action_l = {0: 'ACCELERATE', 1: 'DECELERATE'}
        # 'target', 0: change lane, 1: not change lane
        # 'lag', 0: accelerate, 1: decelerate
        if target_action == 0:
            Acc_target_t = max(Acc_target_t, 0) # ensure it is positive
            target_new_v_acc = min(MAX_V_ACC, Acc_target_t, max((MAX_V_VEL-target_vehicle.v_vel)/T, 0))
            target_new_yaw = self.yaw + _sign * MAX_DELTA_YAW
            print(f'Target chose LANE CHANGE, predict lag chose {id2action_l[lag_action]}')
        elif target_action == 1:
            Acc_target_t = max(Acc_target_t, 0) # ensure it is positive
            target_new_v_acc = min(MAX_V_ACC, Acc_target_t, max((MAX_V_VEL-target_vehicle.v_vel)/T, 0))
            target_new_yaw = min(0.0, self.yaw - _sign * MAX_DELTA_YAW)
            print(f'Target choose NOT LANE CHANGE, predict lag chose {id2action_l[lag_action]}')
        if lag_action == 0:
            lag_new_v_acc = min(max(side_lag_vehicle.v_acc, 0) + 1.0, MAX_V_ACC, max((MAX_V_VEL-side_lag_vehicle.v_vel)/T, 0))
            lag_new_yaw = 0.0
            print(f'Lag choose ACCELERATE, predict target chose {id2action_t[target_action]}')
        elif lag_action == 1:
            print('acc_target_lag:', Acc_target_l)
            Acc_target_l = min(Acc_target_l, 0) # ensure it is negative
            lag_new_v_acc = -min(MAX_V_ACC, -Acc_target_l, (side_lag_vehicle.v_vel-1e-6) / T)
            lag_new_yaw = 0.0
            print(f'Lag choose DECELERATE, predict target chose {id2action_t[target_action]}')
        
        return target_new_v_acc, target_new_yaw, lag_new_v_acc, lag_new_yaw



    def write(self, fout):
        for t in self.trajectory:
            frame, loc_x, loc_y, v_vel = t
            fout.write('{},{},{},{},{}\n'.format(self.id, frame, loc_x, loc_y, v_vel))
    
    def __str__(self):
        return '{}'.format(self.id)
        # return 'Vehicle: {}\tFrame: {}\tLoc_X: {:.2f}\tLoc_Y: {:.2f}\tv_Vel: {:.2f}\tv_Acc: {:.2f}\tyaw: {:.2f}\tLane_ID: {}\tstate: {}'.format(self.id, self.frame, self.loc_x, self.loc_y, self.v_vel, self.v_acc, self.yaw, self.lane_id, self.state)
class Lane:
    def __init__(self, _in, _out):
        self._in = _in
        self._out = _out
        self.middle = (_in + _out) / 2

class Environment:
    def __init__(self, lanes):
        self.vehicles = []
        self.lanes = lanes

    def makeGraph(self):
        vehicle_lane_r = []
        vehicle_lane_l = []
        for v in self.vehicles:
            if v.lane_id == 0:
                vehicle_lane_r.append(v)
            else:
                vehicle_lane_l.append(v)
        vehicle_lane_r.sort(key = lambda vehicle: vehicle.loc_y, reverse=True)
        vehicle_lane_l.sort(key = lambda vehicle: vehicle.loc_y, reverse=True)
        for i in range(1, len(vehicle_lane_r)):
            vehicle_lane_r[i].lead_vehicle = vehicle_lane_r[i-1]
        for i in range(1, len(vehicle_lane_l)):
            vehicle_lane_l[i].lead_vehicle = vehicle_lane_l[i-1]
        
        temp_vehicles = sorted(self.vehicles, key = lambda vehicle: vehicle.loc_y, reverse=True)

        for i in range(1, len(temp_vehicles)):
            if temp_vehicles[i].lane_id == temp_vehicles[i-1].lane_id:
                temp_vehicles[i].side_lead_vehicle = temp_vehicles[i-1].side_lead_vehicle
            else:
                temp_vehicles[i].side_lead_vehicle = temp_vehicles[i-1]
        for i in range(len(temp_vehicles)-2, -1, -1):
            if temp_vehicles[i].lane_id == temp_vehicles[i+1].lane_id:
                temp_vehicles[i].side_lag_vehicle = temp_vehicles[i+1].side_lag_vehicle
            else:
                temp_vehicles[i].side_lag_vehicle = temp_vehicles[i+1]

    def SetVechicles(self, vehicles):
        self.vehicles = vehicles
    
    def write(self, output_file):
        fout = open(output_file, 'w')
        fout.write('Vehicle_ID,Frame_ID,Local_X,Local_Y,v_Vel\n')
        for vehicle in self.vehicles:
            vehicle.write(fout)
        fout.close()

    def __str__(self):
        header = '{:8} | {:8} | {:8} | {:8} | {:8} | {:8} | {:8} | {:8} | {:8} | {:8} | {:8} | {:8} |\n'\
        .format('Vehicle', 'Frame', 'Loc_X', 'Loc_Y', 'v_Vel', 'v_Acc', 'Yaw', 'Lane_ID', 'State', 'Lead', 'Side Lead', 'Side Lag') 
        header += '-' * 150 + '\n'
        vehicle_info = ''
        for vehicle in self.vehicles:
            vehicle_info += '{:<8} | {:<8} | {:<8.2f} | {:<8.2f} | {:<8.2f} | {:<8.2f} | {:<8.2f} | {:8} | {:8} | {:8} | {:8} | {:8} |\n'\
            .format(vehicle.id, vehicle.frame, vehicle.loc_x, vehicle.loc_y, vehicle.v_vel, vehicle.v_acc, vehicle.yaw, vehicle.lane_id, vehicle.state, str(vehicle.lead_vehicle), str(vehicle.side_lead_vehicle), str(vehicle.side_lag_vehicle))
        return header + vehicle_info + '=' * 150 + '\n'

if __name__ == '__main__':
    
    trajectory = pd.read_csv('trajectory_post.csv')
    N_FRAME = 100
    # A: 1103, B: 1121, C: 1096, D: 1084, E: 1119
    # ||     |  D  ||
    # ||  E  |     ||
    # ||     |  A  ||
    # ||  B  |     ||
    # ||     |  C  ||
    # A is going to change lane to in front of B

    trajectory_A = trajectory[trajectory.Vehicle_ID == 1103]
    trajectory_B = trajectory[trajectory.Vehicle_ID == 1121]
    trajectory_C = trajectory[trajectory.Vehicle_ID == 1096]
    trajectory_D = trajectory[trajectory.Vehicle_ID == 1084]
    trajectory_E = trajectory[trajectory.Vehicle_ID == 1119]
    env = Environment([Lane(_in=0, _out=15), Lane(_in=0, _out=-15)])
    vehicle_A = Vehicle(predefined_trajectory=trajectory_A, _id=1103, env=env)
    vehicle_B = Vehicle(predefined_trajectory=trajectory_B, _id=1121, env=env)
    vehicle_C = Vehicle(predefined_trajectory=trajectory_C, _id=1096, env=env)
    vehicle_D = Vehicle(predefined_trajectory=trajectory_D, _id=1084, env=env)
    vehicle_E = Vehicle(predefined_trajectory=trajectory_E, _id=1119, env=env)
    env.SetVechicles([vehicle_A, vehicle_B, vehicle_C, vehicle_D, vehicle_E]) # vehicle_B, vehicle_C, vehicle_D, vehicle_E
    # env.makeGraph()
    print('='*150 + '\n' + 'Start Simulation' + '\n' + '='*150)
    for i in range(N_FRAME):
        env.makeGraph()
        # invoke lane change
        if i == 41:
            vehicle_A.state = vehicle_A.LANE_CHANGE
            vehicle_A.target_lane_id = 1 if vehicle_A.lane_id == 0 else 0
            if METHOD == 'game_theory':
                vehicle_B.state = vehicle_B.LANE_CHANGE_LAG
        for vehicle in env.vehicles:
            vehicle.nextStep(env)
        for vehicle in env.vehicles:
            vehicle.update()
        dist = ((vehicle_A.loc_x - vehicle_B.loc_x) ** 2 + (vehicle_A.loc_y - vehicle_B.loc_y) ** 2) ** 0.5
        print('vehicle A, B dist:', dist)

        print(env, end='')
        for vehicle in env.vehicles:
            assert vehicle.v_vel > 0
    print('='*150 + '\n' + 'Simulation Completed' + '\n' + '='*150)
    print('Total num frame: {}'.format(i+1))
    success = True if vehicle_A.lane_id == 1 else False 
    print(f'Vehicle A change lane: {success}')
    env.write(output_path)
