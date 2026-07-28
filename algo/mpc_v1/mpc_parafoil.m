function [u0, Uopt] = mpc_parafoil(x, what, uPrev, Uwarm, P)
% MPC_PARAFOIL  One receding-horizon solve.
%
% Decision variables: heading-rate sequence U = [u_0; ...; u_{N-1}].
% The horizon shrinks to the remaining flight time so that near the
% ground the optimizer is literally placing the touchdown point.
% Prediction uses the ESTIMATED wind held constant; the mismatch with
% the true sheared/gusty wind is absorbed by replanning every cycle.

% Shrinking horizon: never plan past touchdown
tgo = x(3) / P.Vv;
N   = min(P.N, max(3, ceil(tgo / P.Ts)));

% Warm start: shift previous solution one step, pad/trim to length N
U0 = [Uwarm(2:end); Uwarm(end)];
if numel(U0) >= N
    U0 = U0(1:N);
else
    U0 = [U0; repmat(U0(end), N - numel(U0), 1)];
end

lb = -P.umax * ones(N, 1);
ub =  P.umax * ones(N, 1);

opts = optimoptions('fmincon', ...
    'Algorithm', 'sqp', ...
    'Display', 'off', ...
    'MaxIterations', 60, ...
    'MaxFunctionEvaluations', 5000);

Jfun = @(U) mpc_cost(U, x, what, uPrev, P);
Uopt = fmincon(Jfun, U0, [], [], [], [], lb, ub, [], opts);
u0   = Uopt(1);
end

% ------------------------------------------------------------------------
function J = mpc_cost(U, x, what, uPrev, P)
% Rolls the model forward under candidate sequence U and scores it.

J     = 0;
xk    = x;
up    = uPrev;
psiUW = atan2(-what(2), -what(1));   % heading that points INTO the wind

for i = 1:numel(U)
    u  = U(i);
    xk = rk4(@(xx) parafoil_dynamics(xx, u, what, P), xk, P.Ts);

    % Running costs: stay near pad, steer gently and smoothly
    d2 = (xk(1) - P.target(1))^2 + (xk(2) - P.target(2))^2;
    J  = J + P.Qpath*d2 + P.Ru*u^2 + P.Rdu*(u - up)^2;

    % Final approach: align heading into the wind below hApproach
    if xk(3) < P.hApproach
        e = xk(4) - psiUW;
        e = atan2(sin(e), cos(e));   % wrap to [-pi, pi]
        J = J + P.Qhead * e^2;
    end

    up = u;
    if xk(3) <= 0
        break;   % predicted touchdown reached inside the horizon
    end
end

% Terminal cost: weight grows as altitude shrinks so accuracy at
% touchdown dominates late in the flight
d2T = (xk(1) - P.target(1))^2 + (xk(2) - P.target(2))^2;
J   = J + P.Qf * (1 + P.cTerm / max(xk(3), 5)) * d2T;
end
