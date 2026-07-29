function x = rk4(f, x, dt)
% RK4  One classical Runge-Kutta 4 step for xdot = f(x).
k1 = f(x);
k2 = f(x + dt/2*k1);
k3 = f(x + dt/2*k2);
k4 = f(x + dt*k3);
x  = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);
end
