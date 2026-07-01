function cfg = cfg_ladmm_gaussian_expsemi_etd2()
% CFG_LADMM_GAUSSIAN_EXPSEMI_ETD2  Like cfg_ladmm_gaussian_expsemi but ETD2.
%
%   ETD2 approximates D_x m as linear in time over each FP interval and
%   integrates exactly against the semigroup weight, eliminating the
%   midpoint bias of ETD1 at large eps.
%
%   Use with discretize_then_optimize_etd2 (x.mx lives at nt+1 integer
%   time nodes instead of nt half-integer nodes).

    cfg            = cfg_ladmm_gaussian_expsemi();
    cfg.name       = 'ladmm_gaussian_expsemi_etd2';
    cfg.projection = @proj_fokker_planck_expsemi_etd2;
end
