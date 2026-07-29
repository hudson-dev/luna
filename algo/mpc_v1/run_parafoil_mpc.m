%% Guided parafoil return-to-pad with nonlinear MPC
% 3-DOF kinematic parafoil model, power-law wind shear + Ornstein-Uhlenbeck
% gusts, online wind estimation, receding-horizon NMPC solved with fmincon.
%
% Requires: Optimization Toolbox (fmincon).
% Files:    simulate_flight.m, mpc_parafoil.m, parafoil_dynamics.m,
%           wind_model.m, rk4.m
%
% Author: Hudson Kim's guided parachute project

clear; clc; close all;

%% ----------------------- Parameters ------------------------------------
% Vehicle
P.Vh        = 7.0;            % horizontal airspeed [m/s]
P.Vv        = 3.5;            % sink rate [m/s]  (glide ratio = Vh/Vv = 2)
P.umax      = 0.35;           % max commanded heading rate [rad/s]

% Wind field (truth, unknown to controller)
P.W10       = 3.0;            % mean wind speed at 10 m altitude [m/s]
P.Wdir      = deg2rad(210);   % direction wind blows TOWARD [rad]
P.shearExp  = 0.14;           % power-law shear exponent
P.tauGust   = 8.0;            % gust correlation time [s]
P.sigmaGust = 0.8;            % gust intensity (std dev) [m/s]

% Simulation / sensing
P.dt        = 0.05;           % plant integration step [s]
P.Tc        = 1.0;            % control (MPC) update period [s]
P.sigGPS    = 0.2;            % ground-velocity measurement noise [m/s]
P.tauEst    = 5.0;            % wind-estimator low-pass time constant [s]

% MPC
P.Ts        = 2.0;            % prediction step [s]
P.N         = 25;             % max horizon length  (=> 50 s lookahead)
P.Qpath     = 0.02;           % running position weight
P.Ru        = 2.0;            % control effort weight
P.Rdu       = 4.0;            % control rate (smoothness) weight
P.Qf        = 1.0;            % terminal position weight (base)
P.cTerm     = 200;            % terminal weight growth ~ (1 + cTerm/h)
P.hApproach = 60;             % below this altitude, align into wind [m]
P.Qhead     = 40;             % upwind heading alignment weight

% Scenario
P.x0        = [650; -450; 500; deg2rad(90)];  % [x; y; h; psi]
P.target    = [0; 0];                          % landing pad (x, y)

%% ----------------------- Single run ------------------------------------
out = simulate_flight(P, 2);
fprintf('Flight time:             %.1f s\n', out.t(end));
fprintf('Touchdown miss distance: %.1f m\n', out.miss);

%% ----------------------- Plots -----------------------------------------
% Ground track
figure('Name', 'Ground track'); hold on; grid on; axis equal;
plot(out.X(1,:), out.X(2,:), 'b', 'LineWidth', 1.2);
plot(P.target(1), P.target(2), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
plot(P.x0(1), P.x0(2), 'ko', 'MarkerFaceColor', 'k');
idx = 1:400:numel(out.t);
quiver(out.X(1,idx), out.X(2,idx), out.W(1,idx), out.W(2,idx), 0.5, ...
    'Color', [0.55 0.55 0.55]);
xlabel('x [m]'); ylabel('y [m]');
legend('path', 'pad', 'start', 'wind', 'Location', 'best');
title(sprintf('Ground track  (miss = %.1f m)', out.miss));

% 3D trajectory
figure('Name', '3D trajectory');
plot3(out.X(1,:), out.X(2,:), out.X(3,:), 'b', 'LineWidth', 1.2);
hold on; grid on;
plot3(P.target(1), P.target(2), 0, 'rp', 'MarkerSize', 14, ...
    'MarkerFaceColor', 'r');
xlabel('x [m]'); ylabel('y [m]'); zlabel('h [m]');
title('Descent trajectory'); view(35, 25);

% Control and wind estimation
figure('Name', 'Control and wind');
subplot(3,1,1);
plot(out.t, out.U, 'LineWidth', 1.1); grid on;
ylabel('u [rad/s]'); title('Commanded heading rate');
yline(P.umax, 'r--'); yline(-P.umax, 'r--');
subplot(3,1,2);
plot(out.t, out.W(1,:), 'b', out.t, out.What(1,:), 'r--', 'LineWidth', 1.1);
grid on; ylabel('W_x [m/s]'); legend('true', 'estimate');
subplot(3,1,3);
plot(out.t, out.W(2,:), 'b', out.t, out.What(2,:), 'r--', 'LineWidth', 1.1);
grid on; ylabel('W_y [m/s]'); xlabel('t [s]');

% Distance to pad vs altitude
figure('Name', 'Range vs altitude');
d = vecnorm(out.X(1:2,:) - P.target);
plot(out.X(3,:), d, 'LineWidth', 1.2); grid on;
set(gca, 'XDir', 'reverse');
xlabel('altitude [m]'); ylabel('distance to pad [m]');
title('Closing on the pad as altitude bleeds off');

%% ----------------------- Monte Carlo over winds (optional) -------------
runMC = false;   % set true to run (takes a few minutes)
if runMC
    Nmc  = 30;
    miss = zeros(Nmc, 1);
    for i = 1:Nmc
        Pi      = P;
        Pi.W10  = 1 + 4*rand;        % 1 to 5 m/s mean wind
        Pi.Wdir = 2*pi*rand;         % random direction
        o       = simulate_flight(Pi, 100 + i);
        miss(i) = o.miss;
        fprintf('MC %2d/%d: W10 = %.1f m/s, miss = %.1f m\n', ...
            i, Nmc, Pi.W10, miss(i));
    end
    figure('Name', 'Monte Carlo');
    histogram(miss, 12); grid on;
    xlabel('miss distance [m]'); ylabel('count');
    title(sprintf('Monte Carlo over winds: median miss = %.1f m', ...
        median(miss)));
end
