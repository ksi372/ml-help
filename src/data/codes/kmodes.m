clc;  % k-modes
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'kmodes_student_profiles.csv';
K = 3;
maxIter = 20;

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    Gender = categorical(["Male";"Female";"Male";"Female";"Male";"Female";"Male";"Female";"Male";"Female"]);
    Branch = categorical(["CSE";"IT";"ECE";"CSE";"IT";"ECE";"CSE";"IT";"ECE";"CSE"]);
    Residence = categorical(["Hostel";"Day";"Hostel";"Day";"Hostel";"Hostel";"Day";"Day";"Hostel";"Day"]);
    Study_Style = categorical(["Night";"Morning";"Night";"Morning";"Night";"Night";"Morning";"Morning";"Night";"Morning"]);
    Attendance_Level = categorical(["High";"Medium";"Low";"High";"Medium";"Low";"High";"Medium";"Low";"High"]);
    Result_Status = categorical(["Pass";"Pass";"Fail";"Pass";"Pass";"Fail";"Pass";"Pass";"Fail";"Pass"]);

    data = table(Gender, Branch, Residence, Study_Style, Attendance_Level, Result_Status);
else
    data = readtable(csvFile);
    % Expected all categorical columns
end

disp(data);

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

X = data{:,:};   % categorical table to cell-like content
[n, m] = size(X);

%% Convert to string matrix for easy comparison
Xstr = strings(n,m);
for i = 1:n
    for j = 1:m
        Xstr(i,j) = string(X{i,j});
    end
end

%% =========================
% INITIALIZE MODES
% ==========================
rng(14);
randIdx = randperm(n, K);
modes = Xstr(randIdx, :);

clusterIdx = zeros(n,1);

%% =========================
% K-MODES ITERATION
% ==========================
for iter = 1:maxIter
    oldClusterIdx = clusterIdx;

    % Assignment step
    for i = 1:n
        dist = zeros(K,1);
        for k = 1:K
            dist(k) = sum(Xstr(i,:) ~= modes(k,:));   % simple matching distance
        end
        [~, clusterIdx(i)] = min(dist);
    end

    % Update step
    for k = 1:K
        clusterPoints = Xstr(clusterIdx == k, :);
        if ~isempty(clusterPoints)
            for j = 1:m
                vals = unique(clusterPoints(:,j));
                counts = zeros(length(vals),1);
                for v = 1:length(vals)
                    counts(v) = sum(clusterPoints(:,j) == vals(v));
                end
                [~, maxIdx] = max(counts);
                modes(k,j) = vals(maxIdx);
            end
        end
    end

    if isequal(oldClusterIdx, clusterIdx)
        break;
    end
end

%% =========================
% TOTAL COST
% ==========================
totalCost = 0;
for i = 1:n
    totalCost = totalCost + sum(Xstr(i,:) ~= modes(clusterIdx(i),:));
end

fprintf('\nTotal clustering cost = %d\n', totalCost);

disp('Cluster assignments:');
disp(clusterIdx);

disp('Modes of clusters:');
disp(modes);
