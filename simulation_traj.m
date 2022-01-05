T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\trajectory_A.csv');
num_frame = 101;
frameA = 1;
frameB = frameA + num_frame;
frameC = frameB + num_frame;
frameD = frameC + num_frame;
frameE = frameD + num_frame;

scenario = drivingScenario('SampleTime',0.05);
% add road
roadcenters = [0 -70 0; 0 400 0];
road(scenario,roadcenters, 'Lanes', lanespec(2, 'Width', 15));

% add vehicle 1103 A
v = vehicle(scenario,'ClassID',1, 'Name', 'A');
waypoints = table2array(T(frameA:frameA+num_frame-1, 3:4));
speeds = table2array(T(frameA:frameA+num_frame-1, 5));
trajectory(v,waypoints,speeds);

% add vehicle 1121 B
v = vehicle(scenario,'ClassID',1, 'Name', 'B');
waypoints = table2array(T(frameB:frameB+num_frame-1, 3:4));
speeds = table2array(T(frameB:frameB+num_frame-1, 5));
trajectory(v,waypoints,speeds);

% add vehicle 1096 C
v = vehicle(scenario,'ClassID',1, 'Name', 'C');
waypoints = table2array(T(frameC:frameC+num_frame-1, 3:4));
speeds = table2array(T(frameC:frameC+num_frame-1, 5));
trajectory(v,waypoints,speeds);

% add vehicle 1084 D
v = vehicle(scenario,'ClassID',1, 'Name', 'D');
waypoints = table2array(T(frameD:frameD+num_frame-1, 3:4));
speeds = table2array(T(frameD:frameD+num_frame-1, 5));
trajectory(v,waypoints,speeds);

% add vehicle 1119 E
v = vehicle(scenario,'ClassID',1, 'Name', 'E');
waypoints = table2array(T(frameE:frameE+num_frame-1, 3:4));
speeds = table2array(T(frameE:frameE+num_frame-1, 5));
trajectory(v,waypoints,speeds);

drivingScenarioDesigner(scenario)