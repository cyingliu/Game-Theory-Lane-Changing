num_frame = 101;
frameA = 1;

T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\Game-Theory-Lane-Changing\trajectory\safe_10.csv');
waypoints = table2array(T(frameA:frameA+num_frame-1, 3:4));
plot(waypoints(:, 2), waypoints(:, 1), 'LineWidth', 4);
hold on

T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\Game-Theory-Lane-Changing\trajectory\safe_12.csv');
waypoints = table2array(T(frameA:frameA+num_frame-1, 3:4));
plot(waypoints(:, 2), waypoints(:, 1), '--', 'LineWidth', 4);
hold on
% 
T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\Game-Theory-Lane-Changing\trajectory\safe_13.csv');
waypoints = table2array(T(frameA:frameA+num_frame-1, 3:4));
plot(waypoints(:, 2), waypoints(:, 1), 'LineWidth', 4);
hold on
% 
T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\Game-Theory-Lane-Changing\trajectory\safe_15.csv');
waypoints = table2array(T(frameA:frameA+num_frame-1, 3:4));
plot(waypoints(:, 2), waypoints(:, 1), '--', 'LineWidth', 4);
hold off


% T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\Game-Theory-Lane-Changing\trajectory\rule_based.csv');
% waypoints = table2array(T(frameA:frameA+num_frame-1, 3:4));
% plot(waypoints(:, 2), waypoints(:, 1), 'LineWidth', 4);
% hold on
% 
% T = readtable('D:\我的資料-20210125\電機四上\智慧型汽車導論\final project\Game-Theory-Lane-Changing\trajectory\original.csv');
% waypoints = table2array(T(frameA:frameA+201-1, 3:4));
% plot(waypoints(:, 2), waypoints(:, 1), 'LineWidth', 4);
% hold off

set(gca, 'YDir','reverse')
xlabel('Longitudinal move (m)');
ylabel('Lateral move (m)');
legend('Safety distance=10', 'Safety distance=12', 'Safety distance=13', 'Safety distance=15', 'Location','northwest');

title('Trajectory of Target Vehicle')

