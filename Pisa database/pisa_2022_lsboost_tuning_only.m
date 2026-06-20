%% Board 38 - PISA 2022 LSBoost Tuning Only
% Goal:
% Run ONLY the small LSBoost tuning test now.
% This does NOT run the big final model.
%
% After this finishes, look at:
% pisa_2022_lsboost_tuning_only_results/lsboost_tuning_only_results.csv
%
% Required file in the same folder:
% pisa_2022_student_questionnaire.csv

clear; clc; close all;
rng(42);
tic;

%% =========================
% SETTINGS
%% =========================
fileName = "pisa_2022_student_questionnaire.csv";
resultsFolder = fullfile(pwd, "pisa_2022_lsboost_tuning_only_results");

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

targetDomain = "MATH";

% Small tuning run
tuningRows = 80000;          % use 50000 if you want faster
maxPredictors = 595;         % use all usable predictors after cleaning
tuningTestFraction = 0.20;

missingThreshold = 0.60;
maxCategoricalLevels = 120;
useCountryAsPredictor = true;

% LSBoost configurations to compare
configNames = [
    "Fast_40_20_250_lr010"
    "Strong_90_15_550_lr004"
    "Deep_150_10_650_lr003"
    "Wide_120_12_700_lr0035"
    "Smooth_80_20_800_lr003"
    "Aggressive_180_8_500_lr004"
    "VeryDeep_250_8_650_lr0025"
    "UltraSmooth_120_20_1000_lr002"
];

maxSplitsList = [40, 90, 150, 120, 80, 180, 250, 120];
minLeafList   = [20, 15, 10, 12, 20, 8, 8, 20];
treesList     = [250, 550, 650, 700, 800, 500, 650, 1000];
learnRateList = [0.10, 0.04, 0.03, 0.035, 0.03, 0.04, 0.025, 0.02];

fprintf("===== PISA 2022 LSBoost Tuning Only =====\n");
fprintf("Tuning rows: %d\n", tuningRows);
fprintf("Max predictors: %d\n", maxPredictors);
fprintf("Use country as predictor: %d\n\n", useCountryAsPredictor);

%% =========================
% LOAD DATA
%% =========================
T = readtable(fileName);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

fprintf("Original rows: %d\n", height(T));
fprintf("Original columns: %d\n", width(T));

%% =========================
% SAMPLE BEFORE PROCESSING
%% =========================
if height(T) > tuningRows
    idx = randperm(height(T), tuningRows);
    T = T(idx,:);
end

fprintf("Rows sampled: %d\n", height(T));

%% =========================
% CREATE TARGET
%% =========================
pvVars = strings(10,1);
for i = 1:10
    pvVars(i) = "PV" + i + targetDomain;
end

availablePV = pvVars(ismember(pvVars, string(T.Properties.VariableNames)));

if isempty(availablePV)
    error("No plausible value columns found for %s.", targetDomain);
end

T.ScoreTarget = mean(T{:, availablePV}, 2, "omitnan");
T = rmmissing(T, "DataVariables", "ScoreTarget");

fprintf("Rows after target cleaning: %d\n", height(T));

%% =========================
% BUILD PREDICTORS
%% =========================
X = T;

vars = string(X.Properties.VariableNames);

% Remove plausible values to avoid leakage
X(:, startsWith(vars, "PV")) = [];

% Remove target
X.ScoreTarget = [];

vars = string(X.Properties.VariableNames);

% Remove IDs, weights, and technical columns
removeStarts = [
    "CNTSTUID"
    "CNTSCHID"
    "STUDENTID"
    "SCHOOLID"
    "W_FSTUWT"
    "W_FSTURWT"
    "W_FSCHWT"
    "W_SCHGRNRABWT"
    "SENWT"
    "VER_"
    "BOOKID"
    "FORM"
    "TEST"
    "CYC"
];

toRemove = false(size(vars));

for p = removeStarts'
    toRemove = toRemove | startsWith(vars, p);
end

if ~useCountryAsPredictor
    countryVars = ["CNT","CNTRYID","STRATUM","SUBNATIO","OECD"];
    toRemove = toRemove | ismember(vars, countryVars);
end

X(:, toRemove) = [];

fprintf("Columns after removing PVs/IDs/weights: %d\n", width(X));

%% =========================
% CONVERT TEXT TO CATEGORICAL
%% =========================
vars = string(X.Properties.VariableNames);
for v = vars
    col = X.(v);
    if iscellstr(col) || isstring(col)
        X.(v) = categorical(col);
    end
end

%% =========================
% REMOVE BAD COLUMNS
%% =========================
keep = true(1,width(X));

for i = 1:width(X)
    name = X.Properties.VariableNames{i};
    col = X.(name);

    mr = mean(ismissing(col));
    if mr > missingThreshold
        keep(i) = false;
        continue;
    end

    try
        if isnumeric(col) || islogical(col)
            u = unique(col(~ismissing(col)));
            if numel(u) <= 1
                keep(i) = false;
            end

        elseif iscategorical(col)
            col = removecats(col);
            cats = categories(col);

            if numel(cats) <= 1 || numel(cats) > maxCategoricalLevels
                keep(i) = false;
            end

        else
            keep(i) = false;
        end
    catch
        keep(i) = false;
    end
end

X = X(:, keep);
fprintf("Columns after cleaning filters: %d\n", width(X));

%% =========================
% FILL MISSING VALUES
%% =========================
X = fill_missing_predictors(X);

%% =========================
% FEATURE SCREENING
%% =========================
y = T.ScoreTarget;

scores = score_predictors_simple(X, y);

scoreTable = table(string(X.Properties.VariableNames)', scores(:), ...
    'VariableNames', {'Predictor','Score'});
scoreTable = sortrows(scoreTable, "Score", "descend");

writetable(scoreTable, fullfile(resultsFolder, "predictor_screening_scores.csv"));

numKeep = min(maxPredictors, height(scoreTable));
selectedPredictors = scoreTable.Predictor(1:numKeep);
Xsel = X(:, selectedPredictors);

writetable(table(selectedPredictors(:), scoreTable.Score(1:numKeep), ...
    'VariableNames', {'Predictor','ScreeningScore'}), ...
    fullfile(resultsFolder, "selected_predictors.csv"));

fprintf("Selected predictors: %d\n", width(Xsel));

%% =========================
% FINAL TUNING TABLE
%% =========================
ML = Xsel;
ML.ScoreTarget = y;
ML = rmmissing(ML, "DataVariables", "ScoreTarget");

fprintf("Rows used for tuning ML: %d\n", height(ML));
fprintf("Predictors used: %d\n\n", width(ML)-1);

%% =========================
% TRAIN / TEST SPLIT
%% =========================
cv = cvpartition(height(ML), "Holdout", tuningTestFraction);

trainT = ML(training(cv),:);
testT  = ML(test(cv),:);

XTrain = removevars(trainT, "ScoreTarget");
yTrain = trainT.ScoreTarget;

XTest = removevars(testT, "ScoreTarget");
yTest = testT.ScoreTarget;

fprintf("Train rows: %d\n", height(trainT));
fprintf("Test rows : %d\n\n", height(testT));

%% =========================
% LSBOOST TUNING
%% =========================
numConfigs = numel(configNames);

rmseList = zeros(numConfigs,1);
maeList = zeros(numConfigs,1);
r2List = zeros(numConfigs,1);
timeList = zeros(numConfigs,1);

bestPred = [];
bestRMSE = inf;
bestName = "";

fprintf("===== TESTING LSBOOST CONFIGURATIONS =====\n");

for c = 1:numConfigs
    fprintf("\nTesting %d/%d: %s\n", c, numConfigs, configNames(c));
    tStart = tic;

    treeTemplate = templateTree( ...
        "MaxNumSplits", maxSplitsList(c), ...
        "MinLeafSize", minLeafList(c));

    mdl = fitrensemble(XTrain, yTrain, ...
        "Method","LSBoost", ...
        "Learners",treeTemplate, ...
        "NumLearningCycles",treesList(c), ...
        "LearnRate",learnRateList(c));

    yPred = predict(mdl, XTest);

    rmseList(c) = sqrt(mean((yPred-yTest).^2));
    maeList(c) = mean(abs(yPred-yTest));
    r2List(c) = 1 - sum((yPred-yTest).^2) / sum((yTest-mean(yTest)).^2);
    timeList(c) = toc(tStart);

    fprintf("%s | RMSE %.4f | MAE %.4f | R2 %.4f | Time %.2fs\n", ...
        configNames(c), rmseList(c), maeList(c), r2List(c), timeList(c));

    if rmseList(c) < bestRMSE
        bestRMSE = rmseList(c);
        bestName = configNames(c);
        bestPred = yPred;
    end
end

%% =========================
% RESULTS
%% =========================
Results = table(configNames, maxSplitsList(:), minLeafList(:), treesList(:), learnRateList(:), ...
    rmseList, maeList, r2List, timeList, ...
    'VariableNames', {'Config','MaxNumSplits','MinLeafSize','NumTrees','LearnRate','RMSE','MAE','R2','Seconds'});

Results = sortrows(Results, "RMSE");

disp("===== SORTED LSBOOST TUNING RESULTS =====");
disp(Results);

writetable(Results, fullfile(resultsFolder, "lsboost_tuning_only_results.csv"));

%% =========================
% BEST MODEL PLOTS
%% =========================
fig1 = figure("Visible","off");
scatter(yTest, bestPred, 10, "filled");
hold on;
minVal = min([yTest; bestPred]);
maxVal = max([yTest; bestPred]);
plot([minVal maxVal], [minVal maxVal], "r--", "LineWidth", 1.5);
grid on;
xlabel("Actual PISA Math Score");
ylabel("Predicted PISA Math Score");
title("Best LSBoost Tuning Model: " + bestName);
hold off;
saveas(fig1, fullfile(resultsFolder, "best_lsboost_predicted_vs_actual.png"));

residuals = yTest - bestPred;

fig2 = figure("Visible","off");
scatter(bestPred, residuals, 10, "filled");
hold on;
yline(0, "r--", "LineWidth", 1.5);
grid on;
xlabel("Predicted Score");
ylabel("Residual");
title("Residuals: " + bestName);
hold off;
saveas(fig2, fullfile(resultsFolder, "best_lsboost_residuals.png"));

%% =========================
% SUMMARY
%% =========================
fileID = fopen(fullfile(resultsFolder, "lsboost_tuning_only_summary.txt"), "w");

fprintf(fileID, "PISA 2022 LSBoost Tuning Only Summary\n");
fprintf(fileID, "=====================================\n\n");

fprintf(fileID, "Rows used: %d\n", height(ML));
fprintf(fileID, "Predictors used: %d\n", width(ML)-1);
fprintf(fileID, "Train rows: %d\n", height(trainT));
fprintf(fileID, "Test rows: %d\n\n", height(testT));

fprintf(fileID, "Best configuration:\n");
fprintf(fileID, "Config: %s\n", Results.Config(1));
fprintf(fileID, "MaxNumSplits: %d\n", Results.MaxNumSplits(1));
fprintf(fileID, "MinLeafSize: %d\n", Results.MinLeafSize(1));
fprintf(fileID, "NumTrees: %d\n", Results.NumTrees(1));
fprintf(fileID, "LearnRate: %.4f\n", Results.LearnRate(1));
fprintf(fileID, "RMSE: %.4f\n", Results.RMSE(1));
fprintf(fileID, "MAE: %.4f\n", Results.MAE(1));
fprintf(fileID, "R2: %.4f\n\n", Results.R2(1));

fprintf(fileID, "Use this best configuration for the later big cross-validated run.\n");

fclose(fileID);

elapsed = toc;
fprintf("\nDONE. Results saved in folder: %s\n", resultsFolder);
fprintf("Total runtime: %.2f seconds\n", elapsed);

%% ========================================================================
% LOCAL FUNCTIONS
%% ========================================================================

function X = fill_missing_predictors(X)
    for i = 1:width(X)
        name = X.Properties.VariableNames{i};
        col = X.(name);

        if isnumeric(col) || islogical(col)
            med = median(col, "omitnan");
            if isnan(med)
                med = 0;
            end
            X.(name) = fillmissing(col, "constant", med);

        elseif iscategorical(col)
            col = removecats(col);
            cats = categories(col);

            if ~ismember("Missing", cats)
                col = addcats(col, "Missing");
            end

            col(ismissing(col)) = "Missing";
            X.(name) = removecats(col);

        else
            try
                col = categorical(col);
                col = removecats(col);

                cats = categories(col);
                if ~ismember("Missing", cats)
                    col = addcats(col, "Missing");
                end

                col(ismissing(col)) = "Missing";
                X.(name) = removecats(col);
            catch
                X.(name) = zeros(height(X),1);
            end
        end
    end
end

function scores = score_predictors_simple(X, y)
    scores = zeros(width(X),1);

    for i = 1:width(X)
        col = X.(i);

        try
            if isnumeric(col) || islogical(col)
                c = corr(double(col), y, "Rows", "complete");
                if isnan(c)
                    c = 0;
                end
                scores(i) = abs(c);

            elseif iscategorical(col)
                col = removecats(col);
                cats = categories(col);
                overall = mean(y, "omitnan");
                totalVar = var(y, "omitnan");

                if totalVar == 0 || isnan(totalVar)
                    scores(i) = 0;
                else
                    between = 0;
                    n = numel(y);

                    for j = 1:numel(cats)
                        idx = col == cats{j};

                        if sum(idx) > 5
                            groupMean = mean(y(idx), "omitnan");
                            between = between + (sum(idx)/n) * (groupMean - overall)^2;
                        end
                    end

                    scores(i) = between / totalVar;
                end
            else
                scores(i) = 0;
            end
        catch
            scores(i) = 0;
        end
    end
end
