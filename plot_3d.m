num_frame = 101;
frameA = 1;
frameB = frameA + num_frame;
frameC = frameB + num_frame;
frameD = frameC + num_frame;
frameE = frameD + num_frame;

T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\Game-Theory-Lane-Changing\trajectory\baseline.csv');
waypoints = table2array(T(frameA:frameA+num_frame-1, 3:4));
plot3(waypoints(:, 2), waypoints(:, 2), 1:101, '_');
hold on

waypoints = table2array(T(frameB:frameB+num_frame-1, 3:4));
plot3(waypoints(:, 2), waypoints(:, 2), 1:101, '_');
hold on

waypoints = table2array(T(frameC:frameC+num_frame-1, 3:4));
plot3(waypoints(:, 2), waypoints(:, 2), 1:101, '_');
hold on

waypoints = table2array(T(frameD:frameD+num_frame-1, 3:4));
plot3(waypoints(:, 2), waypoints(:, 2), 1:101, '_');
hold on

waypoints = table2array(T(frameE:frameE+num_frame-1, 3:4));
plot3(waypoints(:, 2), waypoints(:, 2), 1:101, '_');
hold off

grid on

view(40, 30);
xlabel('Lateral move (m)');
ylabel('Longitudinal move (m)');
zlabel('Time (ms)');

legend('Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E');
title("Vehicle's Positions Versus Time");
