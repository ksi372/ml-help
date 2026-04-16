clc;
clear;
close all;

%% =========================
%  SETTINGS
% ==========================
useSynthetic = true;   % true = use built-in synthetic dataset
csvFile = 'bodyfat_data.csv';  % used only if useSynthetic = false

%% =========================
%  LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(1);   % for repeatable results
    n = 100;

    Age = randi([18 60], n, 1);
    DailyCalorieIntake = randi([1800 3500], n, 1);
    SleepHours = 5 + 4*rand(n,1);              % between 5 and 9
    WorkoutSessions = randi([0 6], n, 1);

    % Synthetic target with some noise
    BodyFat = 0.25*Age + 0.004*DailyCalorieIntake - 1.8*SleepHours ...
              - 1.5*WorkoutSessions + randn(n,1)*2 + 8;

    data = table(Age, DailyCalorieIntake, SleepHours, WorkoutSessions, BodyFat);
else
    data = readtable(csvFile);

    % Example expected column names:
    % Age, DailyCalorieIntake, SleepHours, WorkoutSessions, BodyFat
end

disp('First 5 rows of dataset:');
disp(head(data));

%% =========================
%  PREPROCESSING
% ==========================
% Remove missing values
data = rmmissing(data);

% Remove duplicate rows
data = unique(data);

% Separate input and target
X = [data.Age, data.DailyCalorieIntake, data.SleepHours, data.WorkoutSessions];
y = data.BodyFat;

% Normalize features
X = normalize(X);

%% =========================
%  TRAIN-TEST SPLIT
% ==========================
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);

X_test = X(test(cv), :);
y_test = y(test(cv), :);

%% =========================
%  MODEL BUILDING
% ==========================
model = fitlm(X_train, y_train);

disp('Linear Regression Model:');
disp(model);

%% =========================
%  PREDICTION
% ==========================
y_pred = predict(model, X_test);

%% =========================
%  EVALUATION
% ==========================
MAE = mean(abs(y_test - y_pred));
MSE = mean((y_test - y_pred).^2);
RMSE = sqrt(MSE);
R2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

fprintf('\nEvaluation Metrics:\n');
fprintf('MAE  = %.4f\n', MAE);
fprintf('MSE  = %.4f\n', MSE);
fprintf('RMSE = %.4f\n', RMSE);
fprintf('R^2  = %.4f\n', R2);

%% =========================
%  VISUALIZATION
% ==========================
figure;
scatter(y_test, y_pred, 'filled');
xlabel('Actual Body Fat');
ylabel('Predicted Body Fat');
title('Actual vs Predicted Body Fat');
grid on;

figure;
plot(y_test, 'b-o', 'LineWidth', 1.5);
hold on;
plot(y_pred, 'r-*', 'LineWidth', 1.5);
legend('Actual', 'Predicted');
title('Actual and Predicted Values');
xlabel('Test Sample');
ylabel('Body Fat');
grid on;
