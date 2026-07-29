function xdot = parafoil_dynamics(x, u, w, P)
% PARAFOIL_DYNAMICS  3-DOF kinematic guidance model.
%
%   x = [px; py; h; psi]   position [m], altitude [m], heading [rad]
%   u = commanded heading rate [rad/s]
%   w = [wx; wy] wind velocity [m/s]
%
% The canopy flies at fixed airspeed P.Vh and sink rate P.Vv; the only
% control authority is turning. Physically u comes from asymmetric brake
% deflection via the coordinated-turn relation u = (g/Vh)*tan(phi).

psi  = x(4);
xdot = [P.Vh*cos(psi) + w(1);
        P.Vh*sin(psi) + w(2);
        -P.Vv;
        u];
end
