function [a_proj, b_proj] = project_parabolic(a, b)
    % Project onto the set {(a,b) : a + b^2/4 <= 0} pointwise
    % This is the set "below" the parabola a = -b^2/4
    %
    % Vectorized implementation for efficiency

    % Check which points are already feasible
    feasible = (a + b.^2/4 <= 1e-14);  % small tolerance for numerical stability

    % Initialize output
    a_proj = a;
    b_proj = b;

    % For infeasible points, project onto boundary a = -b^2/4
    if any(~feasible(:))
        % Get infeasible points
        a_inf = a(~feasible);
        b_inf = b(~feasible);

        % Solve for projection using cubic equation (vectorized)
        % Minimize (a' - a)^2 + (b' - b)^2 subject to a' = -b'^2/4
        % Derivative gives: b'^3 + (4a + 8)b' - 8b = 0
        % Using solve_cubic with coefficients: [1, 0, 4a+8, -8b]

        b_proj_inf = solve_cubic(ones(size(a_inf)), ...
                                  zeros(size(a_inf)), ...
                                  4*a_inf + 8, ...
                                  -8*b_inf);

        % Compute projected a values on the boundary
        a_proj_inf = -b_proj_inf.^2/4;

        % Update output arrays
        a_proj(~feasible) = a_proj_inf;
        b_proj(~feasible) = b_proj_inf;
    end
end
