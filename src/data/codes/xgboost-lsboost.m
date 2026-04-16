clc;  % gradient boosting regression (LSBoost, MATLAB)
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'xgboost_crop_yield.csv';
numLearners = 100;

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(22);
    n = 250;

    Rainfall_mm = randi([650 1100], n, 1);
    Avg_Temperature = 22 + 10*rand(n,1);
    Soil_pH = 5.5 + 1.8*rand(n,1);
    Fertilizer_Used_kg = randi([80 180], n, 1);
    Pesticide_Used_kg = randi([8 25], n, 1);
    Irrigation_Hours = randi([5 18], n, 1);
    Humidity = randi([50 85], n, 1);

    Crop_Yield = 0.0025*Rainfall_mm - 0.08*Avg_Temperature + 0.9*Soil_pH ...
                 + 0.012*Fertilizer_Used_kg - 0.015*Pesticide_Used_kg ...
                 + 0.05*Irrigation_Hours + 0.01*Humidity + randn(n,1)*0.2;

    data = table(Rainfall_mm, Avg_Temperature, Soil_pH, Fertilizer_Used_kg, ...
                 Pesticide_Used_kg, Irrigation_Hours, Humidity, Crop_Yield);
else
    data = readtable(csvFile);
end

disp(head(data));

%% PREPROCESSING
data = rmmissing(data);
data = unique(data);

X = [data.Rainfall_mm, data.Avg_Temperature, data.Soil_pH, ...
     data.Fertilizer_Used_kg, data.Pesticide_Used_kg, ...
     data.Irrigation_Hours, data.Humidity];
y = data.Crop_Yield;

%% TRAIN-TEST SPLIT
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test  = X(test(cv), :);
y_test  = y(test(cv), :);

%% MODEL BUILDING
template = templateTree('MaxNumSplits', 10);
model = fitrensemble(X_train, y_train, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', numLearners, ...
    'Learners', template);

%% PREDICTION
y_pred = predict(model, X_test);

%% EVALUATION
MAE = mean(abs(y_test - y_pred));
MSE = mean((y_test - y_pred).^2);
RMSE = sqrt(MSE);
R2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

fprintf('\nMAE  = %.4f\n', MAE);
fprintf('MSE  = %.4f\n', MSE);
fprintf('RMSE = %.4f\n', RMSE);
fprintf('R^2  = %.4f\n', R2);

figure;
scatter(y_test, y_pred, 'filled');
xlabel('Actual Yield');
ylabel('Predicted Yield');
title('Actual vs Predicted Crop Yield');
grid on;
