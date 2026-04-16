clc;  % PCA
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'pca_customer_behavior.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(25);
    n = 200;

    Time_On_Site = 5 + 20*rand(n,1);
    Pages_Viewed = round(Time_On_Site + randn(n,1)*2 + 3);
    Cart_Additions = round(0.2*Pages_Viewed + randn(n,1));
    Purchase_Frequency = round(0.6*Cart_Additions + randn(n,1) + 1);
    Avg_Order_Value = 100 + 80*Cart_Additions + randn(n,1)*50;
    Discount_Usage = 70 - 0.15*Avg_Order_Value + randn(n,1)*5;
    Return_Rate = 5 + 0.1*Discount_Usage + randn(n,1)*2;
    Review_Count = round(0.8*Purchase_Frequency + randn(n,1));

    data = table(Time_On_Site, Pages_Viewed, Cart_Additions, Purchase_Frequency, ...
                 Avg_Order_Value, Discount_Usage, Return_Rate, Review_Count);
else
    data = readtable(csvFile);
end

disp(head(data));

%% PREPROCESSING
data = rmmissing(data);
data = unique(data);

X = data{:,:};
X = normalize(X);

%% PCA
[coeff, score, latent, ~, explained] = pca(X);

disp('Explained variance (%):');
disp(explained);

%% VISUALIZATION
figure;
pareto(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('PCA Explained Variance');

figure;
scatter(score(:,1), score(:,2), 40, 'filled');
xlabel('PC1');
ylabel('PC2');
title('PCA Projection on First Two Components');
grid on;

disp('PCA Coefficients:');
disp(coeff);
