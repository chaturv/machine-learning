clear;
% data = csvread('./yahoo_data/GE_20150101-20160528.csv', 1, 5);
data = csvread('./yahoo_data/GOOG_20140101-20160528.csv', 1, 5);

prc = data(:,2);
vol = data(:,1);

% vol = zeros(606,1);
% for i=1:605
%     vol(i, 1) = 1000000;
% end   
% vol(606, 1) = 2000000;

%plot(vol)

%sort
% prc = sort(prc);
% vol = sort(vol);

mu_prc = mean(prc);
sigma2_prc = var(prc); % variance
sigma_prc = std(prc);

% normalize - price
prc = (prc - mu_prc) ./ sigma_prc;
mu_prc = mean(prc);
sigma2_prc = var(prc);
sigma_prc = std(prc);

% make Normal dist
pd_prc = makedist('Normal', mu_prc, sigma_prc);
pdf_prc = pdf(pd_prc, prc);

% plot(prc, pdf_prc);

mu_vol = mean(vol);
sigma2_vol = std(vol);
sigma_vol = std(vol);

% normalize - volume
vol = (vol - mu_vol) ./ sigma_vol;
mu_vol = mean(vol);
sigma2_vol = std(vol);
sigma_vol = std(vol);

% make Normal dist
pd_vol = makedist('Normal', mu_vol, sigma_vol);
pdf_vol = pdf(pd_vol, vol);

% plot(vol, pdf_vol);

mu = [mu_prc mu_vol];
%Sigma2 = [sigma2_prc 0; 0 sigma2_vol]; 

% calc co-variance matrix
m = length(prc);
prc_vol = horzcat(prc, vol);
deviations = prc_vol - (ones(m, m) * prc_vol) ./ m;
Sigma2 = (transpose(deviations) * deviations) ./ m;


% ceate axis
% prc_x = linspace(min(prc), max(prc), 50);
% vol_y = linspace(min(vol), max(vol), 50);
 
prc_x = prc;
vol_y = vol;


[PRC, VOL] = meshgrid(prc_x, vol_y);
F = mvnpdf([PRC(:) VOL(:)], mu, Sigma2);
F = reshape(F, length(vol_y), length(prc_x));

hold on;
grid on;
axis auto;

surf(prc_x, vol_y, F);
% scatter3(0, 4, 0.01, 'red', 'LineWidth', 2);

xlabel('Price (normalized)'); ylabel('Vol (normalized)'); zlabel('Probability Density');

hold off;