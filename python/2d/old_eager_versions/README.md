Pre-optimization copies of the 2D solver modules (git HEAD 569f60b, before the
2026-09-24 jit/GPU work), kept here only for side-by-side comparison with
python/2d/setup/. Not imported by anything; safe to delete. Regenerate with

    git show HEAD:python/2d/setup/<file> > python/2d/old_eager_versions/<file>

and diff against the current versions with

    diff -u python/2d/old_eager_versions/projection_2d.py python/2d/setup/projection_2d.py

projection_expsemi_2d.py was never committed (untracked), so its pre-change
form is not recoverable from git; the diff to it was small (pytree
registration, the shared mode solver, one DCT instead of two).
