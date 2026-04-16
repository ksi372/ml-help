clc;
clear;
close all;

%% =========================
%  GRID WORLD SETTINGS
% ==========================
rows = 4;
cols = 4;
numStates = rows * cols;
numActions = 4;   % 1=Up, 2=Down, 3=Left, 4=Right

startState = 1;       % top-left
goalState = 16;       % bottom-right
trapStates = [6 11];  % example trap cells

alpha = 0.1;          % learning rate
gamma = 0.9;          % discount factor
epsilon = 0.2;        % exploration rate
episodes = 500;
maxSteps = 50;

Q = zeros(numStates, numActions);

episodeRewards = zeros(episodes,1);
episodeSteps = zeros(episodes,1);

%% =========================
%  TRAINING
% ==========================
for ep = 1:episodes
    state = startState;
    totalReward = 0;

    for step = 1:maxSteps

        % Epsilon-greedy action selection
        if rand < epsilon
            action = randi(numActions);
        else
            [~, action] = max(Q(state,:));
        end

        % Take action and get next state
        nextState = getNextState(state, action, rows, cols);

        % Reward structure
        if nextState == goalState
            reward = 10;
        elseif ismember(nextState, trapStates)
            reward = -5;
        else
            reward = 0;
        end

        % Q-learning update
        Q(state, action) = Q(state, action) + alpha * ...
            (reward + gamma * max(Q(nextState,:)) - Q(state, action));

        state = nextState;
        totalReward = totalReward + reward;

        % Stop if goal reached
        if state == goalState
            break;
        end
    end

    episodeRewards(ep) = totalReward;
    episodeSteps(ep) = step;
end

%% =========================
%  DISPLAY Q-TABLE
% ==========================
disp('Learned Q-Table:');
disp(Q);

%% =========================
%  LEARNED POLICY
% ==========================
policy = strings(numStates,1);
actionsText = ["Up","Down","Left","Right"];

for s = 1:numStates
    if s == goalState
        policy(s) = "Goal";
    elseif ismember(s, trapStates)
        policy(s) = "Trap";
    else
        [~, bestAction] = max(Q(s,:));
        policy(s) = actionsText(bestAction);
    end
end

disp('Learned Policy:');
for r = 1:rows
    for c = 1:cols
        s = (r-1)*cols + c;
        fprintf('%8s ', policy(s));
    end
    fprintf('\n');
end

%% =========================
%  EVALUATION METRICS
% ==========================
avgReward = mean(episodeRewards);
avgSteps = mean(episodeSteps);

fprintf('\nEvaluation Metrics:\n');
fprintf('Average Reward per Episode = %.4f\n', avgReward);
fprintf('Average Steps per Episode  = %.4f\n', avgSteps);

%% =========================
%  VISUALIZATION
% ==========================
figure;
plot(episodeRewards, 'LineWidth', 1.5);
xlabel('Episode');
ylabel('Total Reward');
title('Reward per Episode');
grid on;

figure;
plot(episodeSteps, 'LineWidth', 1.5);
xlabel('Episode');
ylabel('Steps');
title('Steps per Episode');
grid on;

%% =========================
%  TEST THE LEARNED PATH
% ==========================
disp('Learned Path from Start to Goal:');
state = startState;
path = state;

for k = 1:20
    if state == goalState
        break;
    end
    [~, action] = max(Q(state,:));
    nextState = getNextState(state, action, rows, cols);
    path(end+1) = nextState; %#ok<SAGROW>
    state = nextState;

    if ismember(state, trapStates)
        break;
    end
end

disp(path);

%% =========================
%  FUNCTION
% ==========================
function nextState = getNextState(state, action, rows, cols)
    [r, c] = ind2sub([rows cols], state);

    if action == 1      % Up
        r = max(r-1, 1);
    elseif action == 2  % Down
        r = min(r+1, rows);
    elseif action == 3  % Left
        c = max(c-1, 1);
    elseif action == 4  % Right
        c = min(c+1, cols);
    end

    nextState = sub2ind([rows cols], r, c);
end
