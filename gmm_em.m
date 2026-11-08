clc;
clearvars;
close all;
format long;
rng(67);


function p = gaussianPDF(X, mu, variance)

% creates the Gaussian PDF for observed data X
p = (1 ./ sqrt(2*pi*variance)) .* exp(-((X-mu).^2) ./ (2*variance));

end


%% generate normally distributed data with multiple peaks
K = input(sprintf("\nNumber of Populations:\t\t\t"));

means = zeros(1, K);
vars = zeros(1, K);
sample_counts = zeros(1, K);
x = [];

for k = 1:K
    means(k) = input(sprintf("\nMean of Population %d:\t\t\t", k));
    vars(k) = input(sprintf("Variance of Population %d:\t\t", k));
    sample_counts(k) = input(sprintf("Number of samples in Population %d:\t", k));

    population_k = means(k) + (sqrt(vars(k)) * randn(sample_counts(k), 1));
    x = [x; population_k];
end

true_weights = sample_counts / sum(sample_counts);
N = length(x);
x = x(randperm(N));


%% display the generated data
figure;
histogram(x, 30, 'Normalization', 'pdf', 'FaceColor', [0.7, 0.7, 0.7], 'EdgeColor', 'none');
xlabel('Observation');
ylabel('Probability Density');
title('Original Dataset');
grid on;


%% initialise the Expectation Maximisation algorithm
xs = sort(x);
idx = round(linspace(0.1*N, 0.9*N, K));
mu = xs(idx)';
sigma2 = var(x) * ones(1, K);
weights = ones(1, K) / K;
maxIter = 100;
tolerance = 1e-6;
previousLogLikelihood = -inf;


%% visualisation
xGrid = linspace(min(x)-2, max(x)+2, 500);
figure('Color', 'w');
histogram(x, 30, 'Normalization', 'pdf', 'FaceColor', [0.8, 0.8, 0.8], 'EdgeColor', 'none');

hold on;

colors = lines(K);
hComponents = gobjects(K, 1);

for k = 1:K
    hComponents(k) = plot(xGrid, zeros(size(xGrid)), 'LineWidth', 2.5, 'Color', colors(k,:));
end

hMixture = plot(xGrid, zeros(size(xGrid)), 'k-', 'LineWidth', 2.5);

xlabel('Observation');
ylabel('Probability Density');

legendLabels = cell(1, K+1);
for k = 1:K
    legendLabels{k} = sprintf('Gaussian %d', k);
end
legendLabels{K+1} = 'Combined mixture';

legend([hComponents; hMixture], legendLabels, 'Location', 'northwest');
grid on;

gifFilename = 'GMM_EM_Convergence.gif';
gifDelay = 0.25;


%% expectation maximisation algorithm
for iter = 1:maxIter
    weightedPDF = zeros(N, K);
    for k = 1:K
        weightedPDF(:,k) = weights(k) * gaussianPDF(x, mu(k), sigma2(k));
    end

    responsibilities = weightedPDF ./ sum(weightedPDF, 2);
    Nk = sum(responsibilities, 1);

    weights = Nk / N;
    mu = (responsibilities' * x)' ./ Nk;
    sigma2 = sum(responsibilities .* (bsxfun(@minus, x, mu)).^2, 1) ./ Nk;
    sigma2 = max(sigma2, 1e-4);

    weightedPDF = zeros(N, K);
    for k = 1:K
        weightedPDF(:,k) = weights(k) * gaussianPDF(x, mu(k), sigma2(k));
    end

    logLikelihood = sum(log(max(sum(weightedPDF, 2), realmin)));

    componentCurves = zeros(K, length(xGrid));
    for k = 1:K
        componentCurves(k,:) = weights(k) * gaussianPDF(xGrid', mu(k), sigma2(k))';
        set(hComponents(k), 'YData', componentCurves(k,:));
    end
    mixtureCurve = sum(componentCurves, 1);
    set(hMixture, 'YData', mixtureCurve);
    title(sprintf(['EM Iteration: %d | Log-Likelihood: %.2f\n' 'Weights: %s'], iter, logLikelihood, mat2str(weights, 3)));
    drawnow;

    frame = getframe(gcf);
    imageData = frame2im(frame);
    [indexedImage, colourMap] = rgb2ind(imageData, 256);
    if iter == 1
        imwrite(indexedImage, colourMap, gifFilename, 'gif', 'LoopCount', Inf, 'DelayTime', gifDelay);
    else
        imwrite(indexedImage, colourMap, gifFilename, 'gif', 'WriteMode', 'append', 'DelayTime', gifDelay);
    end
    pause(gifDelay);

    if abs(logLikelihood - previousLogLikelihood) < tolerance
        break;
    end
    previousLogLikelihood = logLikelihood;
end


%% display the final curve estimates
fprintf('\n\nEM converged after %d iterations\n', iter);
for k = 1:K
    fprintf('\nGaussian %d:', k);
    fprintf('\nWeight:\t\t\t\t\t%.4f', weights(k));
    fprintf('\nMean:\t\t\t\t\t%.4f', mu(k));
    fprintf('\nVariance:\t\t\t\t%.4f', sigma2(k));
    fprintf('\nStandard deviation:\t\t\t%.4f\n', sqrt(sigma2(k)));
end