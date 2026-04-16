clc;
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'kmeans_student_grouping.csv';
K = 3;

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(13);
    n = 150;

    CGPA = 5 + 5*rand(n,1);
    Aptitude_Score = randi([40 100], n, 1);
    Coding_Score = randi([35 100], n, 1);
    Attendance = randi([55 100], n, 1);

    data = table(CGPA, Aptitude_Score, Coding_Score, Attendance);
else
    data = readtable(csvFile);
    % Expected columns:
    % CGPA, Aptitude_Score, Coding_Score, Attendance
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

X = [data.CGPA, data.Aptitude_Score, data.Coding_Score, data.Attendance];
X = normalize(X);

%% =========================
% MODEL BUILDING
% ==========================
[idx, C, sumd] = kmeans(X, K, 'Replicates', 5);

inertia = sum(sumd);

fprintf('Inertia = %.4f\n', inertia);

%% =========================
% SILHOUETTE SCORE
% ==========================
s = silhouette(X, idx);
silhouetteScore = mean(s);
fprintf('Silhouette Score = %.4f\n', silhouetteScore);

%% =========================
% VISUALIZATION
% ==========================
figure;
gscatter(X(:,1), X(:,2), idx);
hold on;
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 14, 'LineWidth', 3);
xlabel('Normalized CGPA');
ylabel('Normalized Aptitude Score');
title('K-Means Clusters');
grid on;
