function [w, gust] = wind_model(h, gust, P)
% WIND_MODEL  Mean wind with power-law altitude shear + OU gusts.
%
% Mean profile (atmospheric boundary layer power law):
%   |W(h)| = W10 * (h/10)^shearExp
%
% Gusts: discretized Ornstein-Uhlenbeck process, i.e. noise with memory.
% Correlation time tauGust, stationary std dev sigmaGust. This is a
% lightweight stand-in for the Dryden turbulence model.

Wmag  = P.W10 * (max(h, 2)/10)^P.shearExp;
wmean = Wmag * [cos(P.Wdir); sin(P.Wdir)];

gust = gust*(1 - P.dt/P.tauGust) ...
     + P.sigmaGust*sqrt(2*P.dt/P.tauGust)*randn(2,1);

w = wmean + gust;
end
