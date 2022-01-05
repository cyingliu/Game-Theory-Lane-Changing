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
DELTA = 1e-2
MAX_DELTA_YAW = 5.0
SAFE_DISTANCE = 12.0
MAX_V_ACC = 20.0
MAX_V_VEL = 100.0
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
        self.id = _id
        self.frame = 0
        self.loc_x = self.predefined_trajectory.iloc[0]['Local_X']
        self.loc_y = self.predefined_trajectory.iloc[0]['Local_Y']
        self.v_vel = self.predefined_trajectory.iloc[0]['v_Vel']
        self.v_acc = 0
        self.trajectory = [[self.frame, self.loc_x, self.loc_y, self.v_vel]] # list of [frame, x, y, v]
        self.yaw = 0
        self.state = self.PREDFINED
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
                lead_distance = self.cal_distance(next_loc_x, next_loc_y, self.lead_vehicle)
            else:
                lead_distance = float('inf')
            
            if next_yaw - self.yaw < -1.7 and self.id == 1103: # consider only vehicle A
                self.state = self.LANE_CHANGE
                self.target_lane_id = 1 if self.lane_id == 0 else 0
            elif self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE:
                self.state = self.LANE_CHANGE_LAG
            elif self.frame >= 49:
                self.state = self.LANE_KEEP
            elif lead_distance <= SAFE_DISTANCE:
                self.state = self.FOLLOW
                next_v_acc = -MAX_V_ACC
        
        elif self.state == self.FOLLOW:

            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel()
            if self.lead_vehicle != None:
                lead_distance = self.cal_distance(next_loc_x, next_loc_y, self.lead_vehicle)
            else:
                lead_distance = float('inf')
            next_yaw = 0.0

            if lead_distance <= SAFE_DISTANCE:
                next_v_acc = -min(MAX_V_ACC, next_v_vel / T)
            elif lead_distance >= 3 * SAFE_DISTANCE:
                next_v_acc = min(max(self.v_acc, 0) + 1.0, MAX_V_ACC, (MAX_V_VEL - self.v_vel) / T)
            else:
                next_v_acc = 0.0
            if self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE:
                self.state = self.LANE_CHANGE_LAG

        elif self.state == self.LANE_CHANGE:
            
            # METHOD 1: rule-based
            # next_v_acc, next_yaw = self.rule_based_lane_change()
            # METHOD 2: game theory
            print('*' * 150 + '\n' + '{} starting game'.format(self.id))
            next_v_acc, next_yaw = self.game_theory_lane_change('target',self, self.side_lag_vehicle, self.side_lead_vehicle)
            print('*' * 150)
            
            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel()
            # check collision
            if self.side_lag_vehicle != None:
                side_lag_distance = self.cal_distance(next_loc_x, next_loc_y, self.side_lag_vehicle)
            else:
                side_lag_distance = float('inf')
            if self.side_lead_vehicle != None:
                side_lead_distance = self.cal_distance(next_loc_x, next_loc_y, self.side_lead_vehicle)
            else:
                side_lead_distance = float('inf')
            if side_lag_distance <= SAFE_DISTANCE or side_lead_distance <= SAFE_DISTANCE:
                print('\n' + '*' * 10 + 'CANCEL LANE CHANGE' + '*' * 10 + '\n')
                self.state = self.LANE_KEEP
                self.target_lane_id = None
            elif next_loc_x <= env.lanes[self.target_lane_id]._in:
                self.state = self.LANE_KEEP
                self.lane_id = self.LANE_L
                self.target_lane_id = None
            elif self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE:
                self.state = self.LANE_CHANGE_LAG

        elif self.state == self.LANE_KEEP:
            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel()
            # check collision
            if self.lead_vehicle != None:
                lead_distance = self.cal_distance(next_loc_x, next_loc_y, self.lead_vehicle)
            else:
                lead_distance = float('inf')
            if lead_distance <= SAFE_DISTANCE:
                next_v_acc = -min(MAX_V_ACC, next_v_vel / T)
            else:
                next_v_acc = 0.0
            next_yaw = 2.0 * (env.lanes[self.lane_id].middle - self.loc_x)
            if abs(next_yaw - self.yaw) > MAX_DELTA_YAW:
                if next_yaw > self.yaw:
                    next_yaw = self.yaw + MAX_DELTA_YAW
                else:
                    next_yaw = self.yaw - MAX_DELTA_YAW
            if self.side_lead_vehicle != None and self.side_lead_vehicle.state == self.LANE_CHANGE:
                self.state = self.LANE_CHANGE_LAG
            elif next_loc_x <= env.lanes[self.lane_id].middle + DELTA and next_loc_x >= env.lanes[self.lane_id].middle - DELTA:
                self.state = self.FOLLOW
        elif self.state == self.LANE_CHANGE_LAG:
            print('*' * 150 + '\n' + '{} starting game'.format(self.id))
            next_v_acc, next_yaw = self.game_theory_lane_change('lag', self.side_lead_vehicle, self.side_lead_vehicle.side_lag_vehicle, self.side_lead_vehicle.side_lead_vehicle)
            print('*' * 150)
            next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel()
            if self.side_lead_vehicle ==  None or self.side_lead_vehicle.state != self.LANE_CHANGE:
                self.state = self.FOLLOW

        self.loc_x = next_loc_x
        self.loc_y = next_loc_y
        self.v_vel = next_v_vel
        self.v_acc = next_v_acc
        self.yaw = next_yaw

        self.frame += 1
        self.trajectory.append([self.frame, self.loc_x, self.loc_y, self.v_vel])

    def cal_loc_vel(self):
        next_loc_x = self.loc_x +  math.sin(self.yaw / 180 * math.pi) * (self.v_vel * T + 0.5 * self.v_acc * T**2 )
        next_loc_y = self.loc_y +  math.cos(self.yaw / 180 * math.pi) * (self.v_vel * T + 0.5 * self.v_acc * T**2 )
        next_v_vel = self.v_vel + self.v_acc * T
        return next_loc_x, next_loc_y, next_v_vel

    def cal_distance(self, next_loc_x, next_loc_y, ref_vehicle):
        pred_ref_loc_x = ref_vehicle.loc_x + math.sin(ref_vehicle.yaw / 180 * math.pi) * (ref_vehicle.v_vel * T + 0.5 * ref_vehicle.v_acc * T**2 )
        pred_ref_loc_y = ref_vehicle.loc_y + math.cos(ref_vehicle.yaw / 180 * math.pi) * (ref_vehicle.v_vel * T + 0.5 * ref_vehicle.v_acc * T**2 )
        return ((next_loc_x - pred_ref_loc_x) ** 2 + (next_loc_y - pred_ref_loc_y) ** 2) ** 0.5
        # return ((self.loc_x - ref_vehicle.loc_x) ** 2 + (self.loc_y - ref_vehicle.loc_y) ** 2) ** 0.5
    
    def rule_based_lane_change(self):
        next_v_acc = 2.0
        next_yaw = self.yaw - 2.0
        return next_v_acc, next_yaw

    def game_theory_lane_change(self, position, target_vehicle, side_lag_vehicle, side_lead_vehicle): # position: 'target' or 'lag'
        # Reference: Modeling Lane-Changing Behavior in a Connected Environment: A Game Theory Approach
        
        # For target vehicle
        # Acc_target
        if target_vehicle.side_lag_vehicle != None:
            Acc_target_t = 2 / T ** 2 / math.cos(target_vehicle.yaw * 180 / math.pi)* (SAFE_DISTANCE \
             + side_lag_vehicle.loc_y + side_lag_vehicle.v_vel * T + 0.5 * side_lag_vehicle.v_acc *  T**2 \
             - target_vehicle.loc_y - math.cos(target_vehicle.yaw * 180 / math.pi) * target_vehicle.v_vel * T)
            # Acc_target_t = max(Acc_target, 0) # v_acc_target > Acc_target_t is feasible
        else:
            Acc_target_t = 0
        # Acc_lead
        if target_vehicle.side_lead_vehicle != None:
            Acc_lead_t = 2 / T ** 2 / math.cos(target_vehicle.yaw * 180 / math.pi)* (SAFE_DISTANCE \
             - side_lead_vehicle.loc_y - side_lead_vehicle.v_vel * T - 0.5 * side_lead_vehicle.v_acc *  T**2 \
             + target_vehicle.loc_y + math.cos(target_vehicle.yaw * 180 / math.pi) * target_vehicle.v_vel * T)
            # Acc_lead_t = -min(Acc_target, 0) # v_acc_target < Acc_lead_t is feasible
        if target_vehicle.side_lead_vehicle != None and target_vehicle.lead_vehicle != None:
            delta_v_vel = target_vehicle.side_lead_vehicle.v_vel - target_vehicle.lead_vehicle.v_vel
        else:
            delta_v_vel = 0

        Acc_cf = 0.0

        # For lag vehicle
        if target_vehicle.side_lag_vehicle != None:
            Acc_target_l = 2 / T ** 2 * (-SAFE_DISTANCE \
             + target_vehicle.loc_y + math.cos(target_vehicle.yaw * 180 / math.pi) * (target_vehicle.v_vel * T + 0.5 * target_vehicle.v_acc *  T**2) \
             - side_lag_vehicle.loc_y - side_lag_vehicle.v_vel * T)
            # Acc_target_l = -min(Acc_target_l, 0) # v_acc_lag < Acc_target_l is feasible
        else:
            Acc_target_l = 0
        if target_vehicle.side_lead_vehicle != None and target_vehicle.side_lag_vehicle != None:
            Acc_lead_l = 2 / T ** 2 * (-SAFE_DISTANCE\
                + side_lead_vehicle.loc_y + side_lead_vehicle.v_vel * T + 0.5 * side_lead_vehicle.v_acc * T ** 2\
                - side_lag_vehicle.loc_y - side_lag_vehicle.v_vel * T)
            # Acc_lead_l = -min(Acc_lead_l, 0) # v_acc_lag < Acc_lead_l is feasible
        else:
            Acc_lead_l = 0
        Acc_target_Y = min(Acc_target_l, -MAX_V_ACC)
        Acc_lead_Y = min(Acc_lead_l, -MAX_V_ACC)
        
        #################################################
        # --------------------TODO----------------------#
        #################################################
        a110, a111, a112, a113, e11 = 0, -0.75, 0, 1, 0
        a210, a211, a212, a213, e21 = 0, -0.75, 0, 1, 0
        a120, a121, e12 = 0, 0, 1e-8
        a220, a221, e22 = 0, 0, 1e-8
        b110, b111, d11 = 0, -1, 0
        b120, b121, d12 = 0, 1, 0
        b210, b211, d21 = 0, 1, 0
        b220, b221, d22 = 0, 1, 0
        #################################################

        P11 = a110 + a111 * Acc_target_t  + a112 * Acc_lead_t + a113 * delta_v_vel + e11
        P21 = a210 + a211 * Acc_target_t + a212 * Acc_lead_t + a213 * delta_v_vel + e21
        P12 = a120 + a121 * Acc_cf + e12
        P22 = a220 + a221 * Acc_cf + e22
        Q11 = b110 + b111 * Acc_target_l + d11
        Q12 = b120 + b121 * Acc_lead_l + d12
        Q21 = b210 + b211 * Acc_target_Y + d21
        Q22 = b220 + b221 * Acc_lead_Y + d22
        
        game = nash.Game(np.array([[P11, P12], [P21, P22]]), np.array([[Q11, Q12], [Q21, Q22]]))
        equilibria = game.vertex_enumeration()
        eqs = []
        for eq in equilibria:
            eqs.append(eq)
        
        chosen_eq = random.choice(eqs)
       
        if position == 'target':
            strategy = chosen_eq[0]
        else: # 'lag'
            strategy = chosen_eq[1]
        action = np.random.choice(np.arange(2), p=softmax(strategy))
        # 'target', 0: change lane, 1: not change lane
        # 'lag', 0: accelerate, 1: decelerate
        if position == 'target':
            if action == 0:
                Acc_target_t = max(Acc_target_t, 0) # ensure it is positive
                new_v_acc = min(MAX_V_ACC, Acc_target_t)
                new_yaw = self.yaw - MAX_DELTA_YAW
                print('Target chose LANE CHANGE')
            elif action == 1:
                new_v_acc = 0.0
                new_yaw = min(0.0, self.yaw + MAX_DELTA_YAW)
                print('Target choose NOT LANE CHANGE')
        else: # lag
            if action == 0:
                new_v_acc = side_lag_vehicle.v_acc + 1.0
                new_yaw = 0.0
                print('Lag choose ACCELERATE')
            elif action == 1:
                Acc_target_l = min(Acc_target_l, 0) # ensure it is negative
                next_loc_x, next_loc_y, next_v_vel = self.cal_loc_vel()
                print('lag v vel', side_lag_vehicle.v_vel, T)
                new_v_acc = -min(MAX_V_ACC, -Acc_target_l, next_v_vel / T)
                new_yaw = 0.0
                print('Lag choose DECELERATE')
        return new_v_acc, new_yaw



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
            assert vehicle.v_vel > 0
        return header + vehicle_info + '=' * 150 + '\n'

if __name__ == '__main__':
    output_path = 'trajectory_A.csv'
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
    env.SetVechicles([vehicle_A, vehicle_B, vehicle_C, vehicle_D, vehicle_E])
    # env.makeGraph()
    print('='*150 + '\n' + 'Start Simulation' + '\n' + '='*150)
    for i in range(N_FRAME):
        env.makeGraph()
        vehicle_A.nextStep(env)
        vehicle_B.nextStep(env)
        vehicle_C.nextStep(env)
        vehicle_D.nextStep(env)
        vehicle_E.nextStep(env)
        print(env, end='')
    print('='*150 + '\n' + 'Simulation Completed' + '\n' + '='*150)
    print('Total num frame: {}'.format(i+1))
    env.write(output_path)
