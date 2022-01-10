# Game-Theory-Lane-Changing  
- Implement two-player game theory lane chaning with python, and visulaize with MATLAB Automated Driving Toolbox.
- The predinfined routes are extracted from NGSIM dataset, and refer to [Multi-Player Dynamic Game-Based
Automatic Lane-Changing Decision
Model under Mixed Autonomous Vehicle
and Human-Driven Vehicle Environment](https://journals.sagepub.com/doi/full/10.1177/0361198120940990).
- The game theory model and payoff functions refer to [Modeling Lane-Changing Behavior in a Connected Environment:
A Game Theory Approach](https://www.sciencedirect.com/science/article/pii/S2352146515000903).
## Usage
1. Genrate game theoretic simulation trajectory file.
```
# 'trajectory_post.csv' contains predefined routes of vehicles, and are extracted from NGSIM dataset
# use simulation.py to simulate the trajectory of vehicles 
# where the target vehicle interacted with the lag vehicle with the predifined two-player game when performing lane channging
# output 'trajectory_A.csv' for further visualization using MATLAB
python simulation.py
```
2. Visualize the trajectory file.
```
compile and run simulation_traj.m
```
3. Analyze the lane changing trajectory
```
compile and run plot_3d.m, plot_traj.m
```
## Simulation Result
1. In ```DrivingScenario``` directory,  there are driving scenarios files that can be visualized by MATLAB automated driving toolbox.  
- ```rule_based.mat```: Vehicle A based on rule-based lane changing, and fails because of safety distance
- ```data_based.mat```:  The visualization of real world data from NGSIM data, vehicle A fails to change lane.
- ```game_theory.mat```: Vehicle A succeed to change lane via game with Vehicle B, while considering safety distance.
2. In ```video``` directory, there are videos of the visualization of the three driving scenario files.  
