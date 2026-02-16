% Check convergence behavior
% Run this after your main script

figure;
subplot(2,1,1);
semilogy(outs_admm.residual_diff(1:min(end,1000)));
title('Residual (first 1000 iters)');
xlabel('Iteration'); ylabel('||u_{k+1} - u_k||');
grid on;

subplot(2,1,2);
semilogy(outs_admm.residual_diff);
title('Residual (all iterations)');
xlabel('Iteration'); ylabel('||u_{k+1} - u_k||');
grid on;

fprintf('Final residual: %.2e\n', outs_admm.residual_diff(end));
fprintf('Min residual: %.2e (at iter %d)\n', ...
    min(outs_admm.residual_diff), find(outs_admm.residual_diff == min(outs_admm.residual_diff), 1));

% Check if residual is still decreasing or plateaued
last_100 = outs_admm.residual_diff(end-99:end);
if std(last_100)/mean(last_100) < 0.01
    fprintf('STATUS: Residual has PLATEAUED (std/mean < 1%%)\n');
else
    fprintf('STATUS: Residual still CHANGING\n');
end

% Check the transport quality
figure;
subplot(2,2,1);
plot(xx, rho_admm(1,:), 'b', xx, rho0, 'r--');
title('t=0'); legend('computed', 'target');

subplot(2,2,2);
plot(xx, rho_admm(round(nt/2),:));
title('t=0.5 (should be centered at x=0.5)');

subplot(2,2,3);
plot(xx, rho_admm(end,:), 'b', xx, rho1, 'r--');
title('t=1'); legend('computed', 'target');

subplot(2,2,4);
velocity = mx_admm ./ (rho_admm + 1e-10);
imagesc(xx, linspace(0,1,nt), velocity);
colorbar; title('Velocity field v=m/\rho');
xlabel('x'); ylabel('t');
