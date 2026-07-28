function out = simulate_flight(P, seed)
% SIMULATE_FLIGHT  One guided descent from release to touchdown.
%
% The plant integrates the true dynamics under the true (sheared, gusty)
% wind at a fast rate P.dt. The controller only sees noisy GPS ground
% velocity, estimates the wind with a low-pass filter, and re-solves the
% MPC every P.Tc seconds.

rng(seed);

x      = P.x0;            % [px; py; h; psi]
gust   = [0; 0];          % gust state (true wind)
what   = [0; 0];          % wind estimate (controller's belief)
u      = 0;               % current heading-rate command
Uwarm  = zeros(P.N, 1);   % warm-start sequence for the MPC
ctimer = 0;               % time until next MPC solve

kmax = ceil(1.5 * (x(3)/P.Vv) / P.dt);
X    = zeros(4, kmax);
U    = zeros(1, kmax);
W    = zeros(2, kmax);
What = zeros(2, kmax);
T    = zeros(1, kmax);

t = 0; k = 0;
while x(3) > 0 && k < kmax
    k = k + 1;

    % --- True wind at current altitude (unknown to the controller) -----
    [w, gust] = wind_model(x(3), gust, P);

    % --- Sensing: GPS ground velocity with noise ------------------------
    vAir = [P.Vh*cos(x(4)); P.Vh*sin(x(4))];
    vGPS = vAir + w + P.sigGPS*randn(2,1);

    % --- Wind estimation: LPF of (ground velocity - air velocity) -------
    wraw = vGPS - vAir;
    a    = P.dt / (P.dt + P.tauEst);
    what = what + a*(wraw - what);

    % --- MPC update at the control rate ---------------------------------
    if ctimer <= 0
        [u, Uwarm] = mpc_parafoil(x, what, u, Uwarm, P);
        ctimer = P.Tc;
    end
    ctimer = ctimer - P.dt;

    % --- Plant integration (RK4, true wind) -----------------------------
    x = rk4(@(xx) parafoil_dynamics(xx, u, w, P), x, P.dt);
    t = t + P.dt;

    X(:,k) = x;  U(k) = u;  W(:,k) = w;  What(:,k) = what;  T(k) = t;
end

out.X    = X(:, 1:k);
out.U    = U(1:k);
out.W    = W(:, 1:k);
out.What = What(:, 1:k);
out.t    = T(1:k);
out.miss = norm(out.X(1:2, end) - P.target);
end
