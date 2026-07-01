function cfg = cfg_ladmm_gaussian_expsemi_nonuniform()
% CFG_LADMM_GAUSSIAN_EXPSEMI_NONUNIFORM  ETD projection on a non-uniform time
%   grid refined near t=1.  Uses proj_fokker_planck_expsemi_nonuniform and
%   discretize_then_optimize_nonuniform.

    cfg            = cfg_ladmm_gaussian_expsemi();
    cfg.name       = 'ladmm_gaussian_expsemi_nonuniform';
    cfg.projection = @proj_fokker_planck_expsemi_nonuniform;
    cfg.prox_ke    = @prox_ke_cc_nonuniform;
    cfg.tgrid_fn   = @make_nonuniform_tgrid;   % swap for make_uniform_tgrid or @(n,e) make_nonuniform_tgrid(n,e,2)
end
