%% Board 38 - PISA 2022 Quick 10-Model Tournament
% Goal: test 10 regression models on a smaller PISA sample, quickly.
% Use this to choose the best model/settings before the long final run.
%
% Required file in the same folder:
% pisa_2022_student_questionnaire.csv
%
% Required product:
% Statistics and Machine Learning Toolbox

clear; clc; close all;
rng(42);
tic;

%% =========================
% SETTINGS
%% =========================
fileName = "pisa_2022_student_questionnaire.csv";
resultsFolder = fullfile(pwd, "pisa_2022_10model_results");

if ~isfolder(resultsFolder)
    mkdir(resultsFolder);
end

targetDomain = "MATH";

% Small/medium run for testing many models fast
maxRows = 50000;          % use 30000 if too slow, 100000 if your PC is fine
maxPredictors = 250;      % use 150 if too slow, 400 if your PC is fine

testFraction = 0.20;      % one holdout split for speed
missingThreshold = 0.60;
maxCategoricalLevels = 100;

% true = best prediction; false = cleaner student-level interpretation
useCountryAsPredictor = true;

fprintf("===== PISA 2022 Quick 10-Model Tournament =====\n");
fprintf("Rows sampled: %d\n", maxRows);
fprintf("Max predictors: %d\n", maxPredictors);
fprintf("Test fraction: %.2f\n", testFraction);
fprintf("Use country as predictor: %d\n\n", useCountryAsPredictor);

%% =========================
% LOAD DATA
%% =========================
T = readtable(fileName);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

fprintf("Original rows: %d\n", height(T));
fprintf("Original columns: %d\n", width(T));

%% =========================
% SAMPLE ROWS
%% =========================
if height(T) > maxRows
    idx = randperm(height(T), maxRows);
    T = T(idx,:);
end

fprintf("Rows sampled for tournament: %d\n", height(T));

%% =========================
% CREATE TARGET FROM PLAUSIBLE VALUES
%% =========================
pvVars = strings(10,1);
for i = 1:10
    pvVars(i) = "PV" + i + targetDomain;
end

availablePV = pvVars(ismember(pvVars, string(T.Properties.VariableNames)));

if isempty(availablePV)
    error("No plausible values found for %s.", targetDomain);
end

T.ScoreTarget = mean(T{:, availablePV}, 2, "omitnan");
T = rmmissing(T, "DataVariables", "ScoreTarget");

fprintf("Rows after removing missing target: %d\n", height(T));

%% =========================
% BUILD PREDICTOR TABLE
%% =========================
X = T;

% Remove all plausible values to avoid leakage
vars = string(X.Properties.VariableNames);
X(:, startsWith(vars, "PV")) = [];

% Remove target
X.ScoreTarget = [];

% Remove IDs, weights, and technical variables
vars = string(X.Properties.VariableNames);

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

fprintf("Selected predictors: %d\n", width(Xsel));

writetable(table(selectedPredictors, scoreTable.Score(1:numKeep), ...
    'VariableNames', {'Predictor','ScreeningScore'}), ...
    fullfile(resultsFolder, "selected_predictors.csv"));

%% =========================
% FINAL ML TABLE
%% =========================
ML = Xsel;
ML.ScoreTarget = y;
ML = rmmissing(ML, "DataVariables", "ScoreTarget");

fprintf("Rows used for ML: %d\n", height(ML));
fprintf("Predictors used for ML: %d\n\n", width(ML)-1);

%% =========================
% TRAIN / TEST SPLIT
%% =========================
cv = cvpartition(height(ML), "Holdout", testFraction);

trainT = ML(training(cv),:);
testT  = ML(test(cv),:);

yTrain = trainT.ScoreTarget;
yTest  = testT.ScoreTarget;

XTrain = removevars(trainT, "ScoreTarget");
XTest  = removevars(testT, "ScoreTarget");

fprintf("Train rows: %d\n", height(trainT));
fprintf("Test rows : %d\n\n", height(testT));

%% =========================
% TRAIN 10 MODELS
%% =========================
modelNames = strings(10,1);
rmseList = nan(10,1);
maeList  = nan(10,1);
r2List   = nan(10,1);
timeList = nan(10,1);

bestPred = [];
bestRMSE = inf;
bestName = "";

fprintf("===== TRAINING 10 MODELS =====\n");

% 1. Regression Tree
m = 1;
modelNames(m) = "Regression Tree";
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrtree(XTrain,yTrain), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 2. Fine Regression Tree
m = 2;
modelNames(m) = "Fine Regression Tree";
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrtree(XTrain,yTrain,"MinLeafSize",5), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 3. Bagged Trees
m = 3;
modelNames(m) = "Bagged Trees";
t = templateTree("MinLeafSize",10);
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrensemble(XTrain,yTrain,"Method","Bag","Learners",t,"NumLearningCycles",100), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 4. Bagged Trees Deep
m = 4;
modelNames(m) = "Bagged Trees Deep";
t = templateTree("MinLeafSize",3);
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrensemble(XTrain,yTrain,"Method","Bag","Learners",t,"NumLearningCycles",100), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 5. LSBoost Fast
m = 5;
modelNames(m) = "LSBoost Fast";
t = templateTree("MaxNumSplits",40,"MinLeafSize",20);
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrensemble(XTrain,yTrain,"Method","LSBoost","Learners",t,"NumLearningCycles",150,"LearnRate",0.10), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 6. LSBoost Strong
m = 6;
modelNames(m) = "LSBoost Strong";
t = templateTree("MaxNumSplits",90,"MinLeafSize",15);
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrensemble(XTrain,yTrain,"Method","LSBoost","Learners",t,"NumLearningCycles",300,"LearnRate",0.05), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 7. LSBoost Deep
m = 7;
modelNames(m) = "LSBoost Deep";
t = templateTree("MaxNumSplits",150,"MinLeafSize",10);
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrensemble(XTrain,yTrain,"Method","LSBoost","Learners",t,"NumLearningCycles",300,"LearnRate",0.03), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 8. Linear Ridge
m = 8;
modelNames(m) = "Linear Ridge";
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrlinear(XTrain,yTrain,"Learner","leastsquares","Regularization","ridge"), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 9. Gaussian SVM
m = 9;
modelNames(m) = "Gaussian SVM";
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrsvm(XTrain,yTrain,"KernelFunction","gaussian","Standardize",true), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

% 10. KNN Regression
m = 10;
modelNames(m) = "KNN Regression";
[rmseList(m), maeList(m), r2List(m), timeList(m), pred] = train_eval_model( ...
    modelNames(m), @() fitrknn(XTrain,yTrain,"NumNeighbors",20,"Standardize",true), XTest, yTest);
[bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmseList(m),modelNames(m),pred);

%% =========================
% RESULTS TABLE
%% =========================
Results = table(modelNames, rmseList, maeList, r2List, timeList, ...
    'VariableNames', {'Model','RMSE','MAE','R2','TrainEvalSeconds'});

Results = sortrows(Results, "RMSE");
disp("===== SORTED RESULTS =====");
disp(Results);

writetable(Results, fullfile(resultsFolder, "ten_model_tournament_results.csv"));

%% =========================
% BEST MODEL FIGURES
%% =========================
if ~isempty(bestPred)
    fig1 = figure("Visible","off");
    scatter(yTest,bestPred,10,"filled");
    hold on;
    minVal = min([yTest; bestPred]);
    maxVal = max([yTest; bestPred]);
    plot([minVal maxVal],[minVal maxVal],"r--","LineWidth",1.5);
    grid on;
    xlabel("Actual PISA Math Score");
    ylabel("Predicted PISA Math Score");
    title("Best Tournament Model: " + bestName);
    hold off;
    saveas(fig1, fullfile(resultsFolder, "best_model_predicted_vs_actual.png"));

    residuals = yTest - bestPred;
    fig2 = figure("Visible","off");
    scatter(bestPred,residuals,10,"filled");
    hold on;
    yline(0,"r--","LineWidth",1.5);
    grid on;
    xlabel("Predicted Score");
    ylabel("Residual");
    title("Residuals: " + bestName);
    hold off;
    saveas(fig2, fullfile(resultsFolder, "best_model_residuals.png"));
end

%% =========================
% SUMMARY
%% =========================
fileID = fopen(fullfile(resultsFolder, "ten_model_summary.txt"), "w");

fprintf(fileID, "PISA 2022 10-Model Tournament Summary\n");
fprintf(fileID, "=====================================\n\n");
fprintf(fileID, "Rows used: %d\n", height(ML));
fprintf(fileID, "Predictors used: %d\n", width(ML)-1);
fprintf(fileID, "Train rows: %d\n", height(trainT));
fprintf(fileID, "Test rows: %d\n\n", height(testT));

fprintf(fileID, "Best model: %s\n", Results.Model(1));
fprintf(fileID, "RMSE: %.4f\n", Results.RMSE(1));
fprintf(fileID, "MAE: %.4f\n", Results.MAE(1));
fprintf(fileID, "R2: %.4f\n\n", Results.R2(1));

fprintf(fileID, "Interpretation:\n");
fprintf(fileID, "- This tournament is for fast model comparison, not final reporting.\n");
fprintf(fileID, "- Use the best model/settings from this run for a larger cross-validated final model.\n");
fprintf(fileID, "- Plausible values, IDs, weights, and technical variables were removed from predictors.\n");

fclose(fileID);

elapsed = toc;
fprintf("\nDONE. Results saved in folder: %s\n", resultsFolder);
fprintf("Total runtime: %.2f seconds\n", elapsed);

%% ========================================================================
% LOCAL FUNCTIONS
%% ========================================================================

function [rmse, mae, r2, seconds, yPred] = train_eval_model(modelName, trainFcn, XTest, yTest)
    rmse = NaN; mae = NaN; r2 = NaN; yPred = [];
    tStart = tic;

    try
        fprintf("Training %s...\n", modelName);
        mdl = trainFcn();
        yPred = predict(mdl, XTest);

        rmse = sqrt(mean((yPred-yTest).^2));
        mae = mean(abs(yPred-yTest));
        r2 = 1 - sum((yPred-yTest).^2) / sum((yTest-mean(yTest)).^2);

        seconds = toc(tStart);

        fprintf("%s | RMSE %.4f | MAE %.4f | R2 %.4f | Time %.2fs\n", ...
            modelName, rmse, mae, r2, seconds);

    catch ME
        seconds = toc(tStart);
        fprintf("%s FAILED: %s\n", modelName, ME.message);
    end
end

function [bestRMSE,bestName,bestPred] = update_best(bestRMSE,bestName,bestPred,rmse,modelName,pred)
    if ~isnan(rmse) && rmse < bestRMSE
        bestRMSE = rmse;
        bestName = modelName;
        bestPred = pred;
    end
end

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
