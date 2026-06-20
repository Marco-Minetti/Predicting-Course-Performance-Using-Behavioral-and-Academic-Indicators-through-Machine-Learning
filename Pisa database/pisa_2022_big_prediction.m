%% Board 38 - PISA 2022 Big Prediction Pipeline
% Goal:
% Use many PISA 2022 student-questionnaire columns to predict student math performance.
%
% Main tasks:
% 1) Regression: predict MathScore from average of PV1MATH ... PV10MATH
% 2) Classification: predict MathLevel = Low / Medium / High
% 3) Feature importance: identify strongest predictors
%
% Required file in the same folder:
% pisa_2022_student_questionnaire.csv
%
% Required MATLAB product:
% Statistics and Machine Learning Toolbox

clear; clc; close all;
rng(42);
tic;

%% =========================
% SETTINGS
%% =========================
fileName = "pisa_2022_student_questionnaire.csv";
resultsFolder = fullfile(pwd, "pisa_2022_big_results_v2");

if ~isfolder(resultsFolder)
    [success,msg] = mkdir(resultsFolder);
    if ~success
        error("Could not create results folder: %s", msg);
    end
end

targetDomain = "MATH";        % "MATH", "READ", or "SCIE"

% Bigger setup:
% If this is too slow, reduce maxRows or maxPredictors.
kFolds = 3;
maxRows = 150000;
maxPredictors = 700;

missingThreshold = 0.60;
maxCategoricalLevels = 120;

% Stronger ensemble settings:
numBagTrees = 180;
numBoostTreesFast = 250;
numBoostTreesStrong = 450;

% Optional country filter:
% "" = use all countries.
% Example: countryFilter = "ITA";
countryFilter = "";

% Keeping country can improve prediction because country/school-system differences are real.
% Set to false if you want a cleaner student-only interpretation.
useCountryAsPredictor = true;

fprintf("===== PISA 2022 Big Prediction Pipeline V2 =====\n");
fprintf("Results folder: %s\n", resultsFolder);
fprintf("Target domain: %s\n", targetDomain);
fprintf("Max rows: %s\n", string(maxRows));
fprintf("Max predictors: %d\n", maxPredictors);
fprintf("K-folds: %d\n", kFolds);
fprintf("Use country as predictor: %d\n\n", useCountryAsPredictor);

%% =========================
% LOAD DATA
%% =========================
T = readtable(fileName);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

fprintf("Original rows: %d\n", height(T));
fprintf("Original columns: %d\n", width(T));

%% =========================
% OPTIONAL COUNTRY FILTER
%% =========================
if countryFilter ~= "" && ismember("CNT", string(T.Properties.VariableNames))
    countryValues = string(T.CNT);
    keepCountry = countryValues == countryFilter;
    T = T(keepCountry,:);
    fprintf("Filtered to country %s. Rows now: %d\n", countryFilter, height(T));
end

%% =========================
% SAMPLE ROWS FOR SPEED
%% =========================
if isfinite(maxRows) && height(T) > maxRows
    idx = randperm(height(T), maxRows);
    T = T(idx,:);
    fprintf("Sampled rows for speed: %d\n", height(T));
end

%% =========================
% CREATE TARGET FROM PLAUSIBLE VALUES
%% =========================
pvVars = strings(10,1);
for i = 1:10
    pvVars(i) = "PV" + i + targetDomain;
end

availablePV = pvVars(ismember(pvVars, string(T.Properties.VariableNames)));

if isempty(availablePV)
    error("No plausible value columns found for %s.", targetDomain);
end

fprintf("\nUsing plausible values as target:\n");
disp(availablePV);

T.ScoreTarget = mean(T{:, availablePV}, 2, "omitnan");
T = rmmissing(T, "DataVariables", "ScoreTarget");

fprintf("Rows after removing missing target: %d\n", height(T));

%% =========================
% BUILD PREDICTOR TABLE
%% =========================
X = T;

% Remove all plausible values to avoid leakage
vars = string(X.Properties.VariableNames);
isPV = startsWith(vars, "PV");
X(:, isPV) = [];

% Remove target
X.ScoreTarget = [];

% Remove IDs, weights, and technical design columns
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
% CONVERT TEXT COLUMNS TO CATEGORICAL
%% =========================
vars = string(X.Properties.VariableNames);
for v = vars
    col = X.(v);
    if iscellstr(col) || isstring(col)
        X.(v) = categorical(col);
    end
end

%% =========================
% REMOVE HIGH-MISSING, CONSTANT, AND BAD COLUMNS
%% =========================
keep = true(1,width(X));

for i = 1:width(X)
    varName = X.Properties.VariableNames{i};
    col = X.(varName);

    mr = mean(ismissing(col));
    if mr > missingThreshold
        keep(i) = false;
        continue;
    end

    try
        if isnumeric(col) || islogical(col)
            nonMissing = col(~ismissing(col));
            u = unique(nonMissing);
            if numel(u) <= 1
                keep(i) = false;
                continue;
            end

        elseif iscategorical(col)
            col = removecats(col);
            cats = categories(col);
            if numel(cats) <= 1 || numel(cats) > maxCategoricalLevels
                keep(i) = false;
                continue;
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
% If columns are below maxPredictors, we use all columns.
% If columns exceed maxPredictors, we rank predictors using simple univariate association.
% This ranking is useful for speed, but final claims should still be treated as predictive associations.

y = T.ScoreTarget;

scores = score_predictors_simple(X, y);

scoreTable = table(string(X.Properties.VariableNames)', scores(:), ...
    'VariableNames', {'Predictor','Score'});
scoreTable = sortrows(scoreTable, "Score", "descend");
writetable(scoreTable, fullfile(resultsFolder, "predictor_screening_scores.csv"));

if width(X) > maxPredictors
    numKeep = maxPredictors;
    selectedPredictors = scoreTable.Predictor(1:numKeep);
    Xsel = X(:, selectedPredictors);
else
    selectedPredictors = string(X.Properties.VariableNames)';
    Xsel = X;
end

selectedTable = table(selectedPredictors(:), ...
    'VariableNames', {'Predictor'});
writetable(selectedTable, fullfile(resultsFolder, "selected_predictors.csv"));

fprintf("Selected predictors: %d\n", width(Xsel));

%% =========================
% CREATE FINAL ML TABLE
%% =========================
ML = Xsel;
ML.ScoreTarget = y;

ML = rmmissing(ML, "DataVariables", "ScoreTarget");

fprintf("\nRows used for ML: %d\n", height(ML));
fprintf("Predictors used for ML: %d\n", width(ML)-1);

%% ============================================================
% TASK 1: REGRESSION - PREDICT PISA SCORE
%% ============================================================
fprintf("\n===== TASK 1: BIG REGRESSION PREDICTION =====\n");

[regResults, regPred, yAll, bestRegModelName] = regression_cv_big_v2( ...
    ML, "ScoreTarget", kFolds, numBagTrees, numBoostTreesFast, numBoostTreesStrong, resultsFolder, "pisa_big_" + lower(targetDomain));

writetable(regResults, fullfile(resultsFolder, "big_regression_results.csv"));

%% ============================================================
% TASK 2: FEATURE IMPORTANCE USING FINAL BAGGED MODEL
%% ============================================================
fprintf("\n===== TASK 2: FEATURE IMPORTANCE =====\n");

[importanceTable, finalBagModel] = regression_feature_importance(ML, "ScoreTarget", numBagTrees, resultsFolder);
writetable(importanceTable, fullfile(resultsFolder, "big_feature_importance.csv"));

disp("Top 30 feature importance values:");
disp(importanceTable(1:min(30,height(importanceTable)),:));

%% ============================================================
% TASK 3: CLASSIFICATION - LOW / MEDIUM / HIGH PERFORMANCE
%% ============================================================
fprintf("\n===== TASK 3: PERFORMANCE LEVEL CLASSIFICATION =====\n");

MLclass = ML;
MLclass.MathLevel = tertile_label(MLclass.ScoreTarget, ["Low","Medium","High"]);
MLclass.ScoreTarget = [];

[classResults, classPred, yClass] = classification_cv_big_v2( ...
    MLclass, "MathLevel", kFolds, numBagTrees, resultsFolder, "pisa_level_" + lower(targetDomain));

writetable(classResults, fullfile(resultsFolder, "performance_level_classification_results.csv"));

%% ============================================================
% FINAL SUMMARY
%% ============================================================
write_summary(resultsFolder, targetDomain, regResults, classResults, width(Xsel), height(ML));

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

function [Results, bestPred, yAll, bestModelName] = regression_cv_big_v2(T, target, k, numBagTrees, numBoostFast, numBoostStrong, resultsFolder, prefix)

    yAll = T.(target);
    cv = cvpartition(height(T), "KFold", k);

    modelNames = [
        "Regression Tree"
        "Bagged Trees"
        "Bagged Trees Deep"
        "LSBoost Fast"
        "LSBoost Strong"
        "LSBoost Deep"
    ];

    numModels = numel(modelNames);

    rmseAll = nan(k,numModels);
    maeAll = nan(k,numModels);
    r2All = nan(k,numModels);
    predOOF = nan(height(T), numModels);

    for fold = 1:k
        trainT = T(training(cv,fold),:);
        testT = T(test(cv,fold),:);

        yTrain = trainT.(target);
        yTest = testT.(target);

        XTrain = removevars(trainT,target);
        XTest = removevars(testT,target);

        models = cell(1,numModels);

        % 1. Regression Tree
        try
            models{1} = fitrtree(XTrain, yTrain);
        catch ME
            fprintf("Fold %d: Regression Tree failed: %s\n", fold, ME.message);
        end

        % 2. Bagged Trees
        try
            t = templateTree("MinLeafSize", 5);
            models{2} = fitrensemble(XTrain, yTrain, ...
                "Method","Bag", ...
                "Learners",t, ...
                "NumLearningCycles",numBagTrees);
        catch ME
            fprintf("Fold %d: Bagged Trees failed: %s\n", fold, ME.message);
        end

        % 3. Deeper Bagged Trees
        try
            t = templateTree("MinLeafSize", 2);
            models{3} = fitrensemble(XTrain, yTrain, ...
                "Method","Bag", ...
                "Learners",t, ...
                "NumLearningCycles",numBagTrees);
        catch ME
            fprintf("Fold %d: Bagged Trees Deep failed: %s\n", fold, ME.message);
        end

        % 4. LSBoost Fast
        try
            t = templateTree("MaxNumSplits", 40, "MinLeafSize", 20);
            models{4} = fitrensemble(XTrain, yTrain, ...
                "Method","LSBoost", ...
                "Learners",t, ...
                "NumLearningCycles",numBoostFast, ...
                "LearnRate",0.10);
        catch ME
            fprintf("Fold %d: LSBoost Fast failed: %s\n", fold, ME.message);
        end

        % 5. LSBoost Strong
        try
            t = templateTree("MaxNumSplits", 80, "MinLeafSize", 15);
            models{5} = fitrensemble(XTrain, yTrain, ...
                "Method","LSBoost", ...
                "Learners",t, ...
                "NumLearningCycles",numBoostStrong, ...
                "LearnRate",0.05);
        catch ME
            fprintf("Fold %d: LSBoost Strong failed: %s\n", fold, ME.message);
        end

        % 6. LSBoost Deep
        try
            t = templateTree("MaxNumSplits", 120, "MinLeafSize", 10);
            models{6} = fitrensemble(XTrain, yTrain, ...
                "Method","LSBoost", ...
                "Learners",t, ...
                "NumLearningCycles",numBoostStrong, ...
                "LearnRate",0.03);
        catch ME
            fprintf("Fold %d: LSBoost Deep failed: %s\n", fold, ME.message);
        end

        for m = 1:numModels
            if isempty(models{m})
                continue;
            end

            yPred = predict(models{m}, XTest);

            predOOF(test(cv,fold),m) = yPred;

            rmseAll(fold,m) = sqrt(mean((yPred-yTest).^2));
            maeAll(fold,m) = mean(abs(yPred-yTest));
            r2All(fold,m) = 1 - sum((yPred-yTest).^2) / sum((yTest-mean(yTest)).^2);
        end

        fprintf("Fold %d complete.\n", fold);
    end

    Results = table;
    Results.Model = modelNames;
    Results.RMSE_Mean = mean(rmseAll,1,"omitnan")';
    Results.RMSE_Std = std(rmseAll,0,1,"omitnan")';
    Results.MAE_Mean = mean(maeAll,1,"omitnan")';
    Results.MAE_Std = std(maeAll,0,1,"omitnan")';
    Results.R2_Mean = mean(r2All,1,"omitnan")';
    Results.R2_Std = std(r2All,0,1,"omitnan")';

    % Remove models that failed all folds
    Results = Results(~isnan(Results.RMSE_Mean), :);

    Results = sortrows(Results, "RMSE_Mean");

    disp(Results);

    bestModelName = Results.Model(1);
    bestIdx = find(modelNames == bestModelName, 1);
    bestPred = predOOF(:,bestIdx);

    fig1 = figure("Visible","off");
    scatter(yAll, bestPred, 8, "filled");
    hold on;
    minVal = min([yAll; bestPred]);
    maxVal = max([yAll; bestPred]);
    plot([minVal maxVal], [minVal maxVal], "r--", "LineWidth", 1.5);
    grid on;
    xlabel("Actual score");
    ylabel("Predicted score");
    title("PISA Big Prediction V2 - Predicted vs Actual (" + bestModelName + ")");
    hold off;
    saveas(fig1, fullfile(resultsFolder, prefix + "_predicted_vs_actual.png"));

    residuals = yAll - bestPred;

    fig2 = figure("Visible","off");
    scatter(bestPred, residuals, 8, "filled");
    hold on;
    yline(0, "r--", "LineWidth", 1.5);
    grid on;
    xlabel("Predicted score");
    ylabel("Residual");
    title("PISA Big Prediction V2 - Residuals (" + bestModelName + ")");
    hold off;
    saveas(fig2, fullfile(resultsFolder, prefix + "_residuals.png"));
end

function [importanceTable, mdl] = regression_feature_importance(T, target, numTrees, resultsFolder)

    y = T.(target);
    X = removevars(T, target);

    t = templateTree("MinLeafSize", 5);

    mdl = fitrensemble(X, y, ...
        "Method","Bag", ...
        "Learners",t, ...
        "NumLearningCycles",numTrees);

    imp = predictorImportance(mdl);
    predictors = string(mdl.PredictorNames);
    predictors = predictors(:);
    imp = imp(:);

    importanceTable = table(predictors, imp, ...
        'VariableNames', {'Predictor','Importance'});

    importanceTable = sortrows(importanceTable, "Importance", "descend");

    topN = min(40, height(importanceTable));

    fig = figure("Visible","off");
    bar(importanceTable.Importance(1:topN));
    xticks(1:topN);
    xticklabels(importanceTable.Predictor(1:topN));
    xtickangle(60);
    ylabel("Predictor importance");
    title("Top PISA Predictors for Score Prediction");
    grid on;
    saveas(fig, fullfile(resultsFolder, "big_feature_importance.png"));
end

function y = tertile_label(score, names3)
    q1 = quantile(score, 1/3);
    q2 = quantile(score, 2/3);

    y = strings(size(score));
    y(score <= q1) = names3(1);
    y(score > q1 & score <= q2) = names3(2);
    y(score > q2) = names3(3);

    y = categorical(y, names3);
end

function [Results, bestPred, yAll] = classification_cv_big_v2(T, target, k, numBagTrees, resultsFolder, prefix)

    yAll = T.(target);
    if ~iscategorical(yAll)
        yAll = categorical(yAll);
        T.(target) = yAll;
    end

    cv = cvpartition(height(T), "KFold", k);

    modelNames = ["Decision Tree", "Bagged Trees", "Bagged Trees Deep"];
    numModels = numel(modelNames);

    accAll = nan(k,numModels);
    f1All = nan(k,numModels);
    predOOF = strings(height(T),numModels);

    for fold = 1:k
        trainT = T(training(cv,fold),:);
        testT = T(test(cv,fold),:);

        yTrain = trainT.(target);
        yTest = testT.(target);

        XTrain = removevars(trainT,target);
        XTest = removevars(testT,target);

        models = cell(1,numModels);

        try
            models{1} = fitctree(XTrain, yTrain);
        catch
        end

        try
            t = templateTree("MinLeafSize", 5);
            models{2} = fitcensemble(XTrain, yTrain, ...
                "Method","Bag", ...
                "Learners",t, ...
                "NumLearningCycles",numBagTrees);
        catch
        end

        try
            t = templateTree("MinLeafSize", 2);
            models{3} = fitcensemble(XTrain, yTrain, ...
                "Method","Bag", ...
                "Learners",t, ...
                "NumLearningCycles",numBagTrees);
        catch
        end

        for m = 1:numModels
            if isempty(models{m})
                continue;
            end

            yPred = predict(models{m}, XTest);

            accAll(fold,m) = mean(yPred == yTest);
            f1All(fold,m) = macro_f1(yTest, yPred);
            predOOF(test(cv,fold),m) = string(yPred);
        end

        fprintf("Fold %d complete.\n", fold);
    end

    Results = table;
    Results.Model = modelNames';
    Results.Accuracy_Mean = mean(accAll,1,"omitnan")';
    Results.Accuracy_Std = std(accAll,0,1,"omitnan")';
    Results.MacroF1_Mean = mean(f1All,1,"omitnan")';
    Results.MacroF1_Std = std(f1All,0,1,"omitnan")';

    Results = Results(~isnan(Results.Accuracy_Mean), :);
    Results = sortrows(Results, "Accuracy_Mean", "descend");

    disp(Results);

    bestModelName = Results.Model(1);
    bestIdx = find(modelNames == bestModelName,1);
    bestPred = categorical(predOOF(:,bestIdx), categories(yAll));

    fig = figure("Visible","off");
    confusionchart(yAll, bestPred);
    title("PISA Performance Level Classification V2 (" + bestModelName + ")");
    saveas(fig, fullfile(resultsFolder, prefix + "_confusion_matrix.png"));
end

function f1 = macro_f1(yTrue, yPred)
    if ~iscategorical(yTrue)
        yTrue = categorical(yTrue);
    end

    if ~iscategorical(yPred)
        yPred = categorical(yPred);
    end

    classes = categories(yTrue);
    f1Scores = zeros(numel(classes),1);

    for i = 1:numel(classes)
        c = classes{i};

        tp = sum(yPred == c & yTrue == c);
        fp = sum(yPred == c & yTrue ~= c);
        fn = sum(yPred ~= c & yTrue == c);

        precision = tp / max(tp+fp, 1);
        recall = tp / max(tp+fn, 1);

        f1Scores(i) = 2*precision*recall / max(precision+recall, 1e-12);
    end

    f1 = mean(f1Scores);
end

function write_summary(resultsFolder, targetDomain, regResults, classResults, numPredictors, numRows)
    fileID = fopen(fullfile(resultsFolder, "pisa_big_summary_report.txt"), "w");

    fprintf(fileID, "PISA 2022 Big Prediction V2 Summary\n");
    fprintf(fileID, "===================================\n\n");

    fprintf(fileID, "Target domain: %s\n", targetDomain);
    fprintf(fileID, "Rows used: %d\n", numRows);
    fprintf(fileID, "Predictors used: %d\n\n", numPredictors);

    fprintf(fileID, "1) Score regression\n");
    fprintf(fileID, "Best model: %s\n", regResults.Model(1));
    fprintf(fileID, "RMSE: %.4f\n", regResults.RMSE_Mean(1));
    fprintf(fileID, "MAE: %.4f\n", regResults.MAE_Mean(1));
    fprintf(fileID, "R2: %.4f\n\n", regResults.R2_Mean(1));

    fprintf(fileID, "2) Performance-level classification\n");
    fprintf(fileID, "Best model: %s\n", classResults.Model(1));
    fprintf(fileID, "Accuracy: %.4f\n", classResults.Accuracy_Mean(1));
    fprintf(fileID, "MacroF1: %.4f\n\n", classResults.MacroF1_Mean(1));

    fprintf(fileID, "Interpretation guidance:\n");
    fprintf(fileID, "- This version uses a larger predictor set and stronger tree ensemble variants.\n");
    fprintf(fileID, "- Plausible values, IDs, and survey weights are removed from predictors to avoid direct leakage.\n");
    fprintf(fileID, "- Keeping country as a predictor may improve prediction but can make country-level differences dominate interpretation.\n");
    fprintf(fileID, "- Results are predictive associations, not causal proof.\n");

    fclose(fileID);
end
