clc;
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'student_performance_data.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(4);
    n = 120;

    Attendance = randi([0 2], n, 1);         % 0=Low,1=Medium,2=High
    Internal_Marks = randi([0 2], n, 1);     % 0=Low,1=Medium,2=High
    Assignment_Status = randi([0 1], n, 1);  % 0=No,1=Yes
    Study_Hours = randi([0 2], n, 1);        % 0=Low,1=Medium,2=High

    Result = strings(n,1);
    for i = 1:n
        score = Attendance(i) + Internal_Marks(i) + Assignment_Status(i) + Study_Hours(i);
        if score >= 4
            Result(i) = "Pass";
        else
            Result(i) = "Fail";
        end
    end

    data = table(Attendance, Internal_Marks, Assignment_Status, Study_Hours, Result);
else
    data = readtable(csvFile);
    % Expected columns:
    % Attendance, Internal_Marks, Assignment_Status, Study_Hours, Result
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

% Convert text categories if present
vars = {'Attendance','Internal_Marks','Assignment_Status','Study_Hours'};
for i = 1:length(vars)
    if iscellstr(data.(vars{i})) || isstring(data.(vars{i})) || iscategorical(data.(vars{i}))
        data.(vars{i}) = grp2idx(categorical(data.(vars{i}))) - 1;
    end
end

data.Result = categorical(data.Result);

X = [data.Attendance, data.Internal_Marks, data.Assignment_Status, data.Study_Hours];
Y = data.Result;

%% =========================
% TRAIN-TEST SPLIT
% ==========================
cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test  = X(test(cv), :);
Y_test  = Y(test(cv), :);

%% =========================
% MODEL BUILDING
% ==========================
model = fitctree(X_train, Y_train, 'SplitCriterion', 'deviance');

%% =========================
% VIEW TREE
% ==========================
view(model, 'Mode', 'graph');

%% =========================
% PREDICTION
% ==========================
Y_pred = predict(model, X_test);

%% =========================
% EVALUATION
% ==========================
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat)) / sum(confMat(:));

disp('Confusion Matrix:');
disp(confMat);
fprintf('\nAccuracy = %.4f\n', accuracy);

classes = categories(Y);
for i = 1:length(classes)
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;

    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2 * precision * recall / (precision + recall + eps);

    fprintf('\nClass: %s\n', string(classes(i)));
    fprintf('Precision = %.4f\n', precision);
    fprintf('Recall    = %.4f\n', recall);
    fprintf('F1-score  = %.4f\n', f1);
end

figure;
confusionchart(Y_test, Y_pred);
title('Decision Tree (ID3-style) Confusion Matrix');
