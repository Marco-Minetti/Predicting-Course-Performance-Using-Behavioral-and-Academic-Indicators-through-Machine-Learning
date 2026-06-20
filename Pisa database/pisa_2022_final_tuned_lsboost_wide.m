%% Board 38 - PISA 2022 Final Tuned LSBoost Wide Model
% Goal:
% Use ONE strong algorithm on a large amount of PISA 2022 data.
%
% Model used:
% LSBoost Regression Ensemble
%
% Why this model:
% In the previous PISA run, LSBoost was the best regression model.
%
% Main output:
% - Cross-validated math score prediction
% - Feature importance
% - Predicted vs actual plot
% - Residual plot
%
% Required file in same folder:
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
resultsFolder = fullfile(pwd, "pisa_2022_final_tuned_lsboost_results");

if ~isfolder(resultsFolder)
    [success,msg] = mkdir(resultsFolder);
    if ~success
        error("Could not create results folder: %s", msg);
    end
end

targetDomain = "MATH";        % "MATH", "READ", or "SCIE"

% Best configuration from the tuning-only run:
% Wide_120_12_700_lr0035
% RMSE = 46.8626, MAE = 36.5998, R2 = 0.7742 on the 80k tuning sample

% Big run settings.
% If it is too slow or crashes, lower maxRows or maxPredictors.
% If you are okay waiting longer, increase them.
kFolds = 3;
maxRows = 600000;             % Try 250000 first. Use Inf for full dataset if your PC can handle it.
maxPredictors = 595;          % Try 900 first. Increase to 1200 if it runs fine.

missingThreshold = 0.60;      % Remove columns with more than 60% missing
maxCategoricalLevels = 120;   % Remove categorical columns with too many categories

% Keep country as predictor?
% true usually improves prediction because country/school-system differences matter.
% false gives cleaner student-level interpretation.
useCountryAsPredictor = true;

% One model only: LSBoost
numBoostTrees = 700;          % More trees = better but slower
learnRate = 0.035;             % Lower = more stable, usually needs more trees
maxNumSplits = 120;            % Bigger trees capture more complex patterns
minLeafSize = 12;             % Lower = more flexible, higher = faster/smoother

fprintf("===== PISA 2022 Final Tuned LSBoost Wide Model =====\n");
fprintf("Results folder: %s\n", resultsFolder);
fprintf("Target domain: %s\n", targetDomain);
fprintf("Max rows: %s\n", string(maxRows));
fprintf("Max predictors: %d\n", maxPredictors);
fprintf("K-folds: %d\n", kFolds);
fprintf("Model: LSBoost only\n\n");

%% =========================
% LOAD DATA
%% =========================
T = readtable(fileName);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

fprintf("Original rows: %d\n", height(T));
fprintf("Original columns: %d\n", width(T));

%% =========================
% SAMPLE ROWS FOR SPEED / MEMORY
%% =========================
if isfinite(maxRows) && height(T) > maxRows
    idx = randperm(height(T), maxRows);
    T = T(idx,:);
    fprintf("Sampled rows: %d\n", height(T));
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

% Simplified ML target:
% Average of PV1MATH ... PV10MATH.
% For official PISA analysis, plausible values are normally analyzed separately.
% For this ML project, this average is a practical prediction target.
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
% We select the strongest predictors using simple univariate association.
% This is mainly for speed and memory control.
y = T.ScoreTarget;

scores = score_predictors_simple(X, y);

scoreTable = table(string(X.Properties.VariableNames)', scores(:), ...
    'VariableNames', {'Predictor','Score'});
scoreTable = sortrows(scoreTable, "Score", "descend");

writetable(scoreTable, fullfile(resultsFolder, "predictor_screening_scores.csv"));

if width(X) > maxPredictors
    selectedPredictors = scoreTable.Predictor(1:maxPredictors);
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
% FINAL ML TABLE
%% =========================
ML = Xsel;
ML.ScoreTarget = y;
ML = rmmissing(ML, "DataVariables", "ScoreTarget");

fprintf("\nRows used for ML: %d\n", height(ML));
fprintf("Predictors used for ML: %d\n", width(ML)-1);

%% ===================================================================
% ONE-MODEL CROSS VALIDATION
%% ===================================================================
fprintf("\n===== LSBOOST CROSS-VALIDATION =====\n");

cv = cvpartition(height(ML), "KFold", kFolds);

rmseList = zeros(kFolds,1);
maeList = zeros(kFolds,1);
r2List = zeros(kFolds,1);

yAll = ML.ScoreTarget;
predOOF = zeros(height(ML),1);

for fold = 1:kFolds
    fprintf("\nTraining fold %d of %d...\n", fold, kFolds);

    trainT = ML(training(cv,fold),:);
    testT  = ML(test(cv,fold),:);

    yTrain = trainT.ScoreTarget;
    yTest  = testT.ScoreTarget;

    XTrain = removevars(trainT, "ScoreTarget");
    XTest  = removevars(testT, "ScoreTarget");

    treeTemplate = templateTree( ...
        "MaxNumSplits", maxNumSplits, ...
        "MinLeafSize", minLeafSize);

    mdl = fitrensemble(XTrain, yTrain, ...
        "Method","LSBoost", ...
        "Learners",treeTemplate, ...
        "NumLearningCycles",numBoostTrees, ...
        "LearnRate",learnRate);

    yPred = predict(mdl, XTest);

    predOOF(test(cv,fold)) = yPred;

    rmseList(fold) = sqrt(mean((yPred-yTest).^2));
    maeList(fold)  = mean(abs(yPred-yTest));
    r2List(fold)   = 1 - sum((yPred-yTest).^2) / sum((yTest-mean(yTest)).^2);

    fprintf("Fold %d | RMSE %.4f | MAE %.4f | R2 %.4f\n", ...
        fold, rmseList(fold), maeList(fold), r2List(fold));
end

%% =========================
% RESULTS
%% =========================
Results = table;
Results.Model = "Final Tuned LSBoost Wide";
Results.Rows_Used = height(ML);
Results.Predictors_Used = width(ML)-1;
Results.KFolds = kFolds;
Results.NumTrees = numBoostTrees;
Results.LearnRate = learnRate;
Results.MaxNumSplits = maxNumSplits;
Results.MinLeafSize = minLeafSize;
Results.RMSE_Mean = mean(rmseList);
Results.RMSE_Std = std(rmseList);
Results.MAE_Mean = mean(maeList);
Results.MAE_Std = std(maeList);
Results.R2_Mean = mean(r2List);
Results.R2_Std = std(r2List);

disp(Results);

writetable(Results, fullfile(resultsFolder, "final_tuned_lsboost_results.csv"));

%% =========================
% FIGURES
%% =========================
fig1 = figure("Visible","off");
scatter(yAll, predOOF, 8, "filled");
hold on;
minVal = min([yAll; predOOF]);
maxVal = max([yAll; predOOF]);
plot([minVal maxVal], [minVal maxVal], "r--", "LineWidth", 1.5);
grid on;
xlabel("Actual PISA Math Score");
ylabel("Predicted PISA Math Score");
title("PISA One-Model LSBoost - Predicted vs Actual");
hold off;
saveas(fig1, fullfile(resultsFolder, "final_predicted_vs_actual.png"));

residuals = yAll - predOOF;

fig2 = figure("Visible","off");
scatter(predOOF, residuals, 8, "filled");
hold on;
yline(0, "r--", "LineWidth", 1.5);
grid on;
xlabel("Predicted Score");
ylabel("Residual");
title("PISA One-Model LSBoost - Residuals");
hold off;
saveas(fig2, fullfile(resultsFolder, "final_residuals.png"));

%% ===================================================================
% TRAIN FINAL MODEL ON ALL DATA FOR FEATURE IMPORTANCE
%% ===================================================================
fprintf("\nTraining final LSBoost model on all data for feature importance...\n");

XFinal = removevars(ML, "ScoreTarget");
yFinal = ML.ScoreTarget;

treeTemplate = templateTree( ...
    "MaxNumSplits", maxNumSplits, ...
    "MinLeafSize", minLeafSize);

finalModel = fitrensemble(XFinal, yFinal, ...
    "Method","LSBoost", ...
    "Learners",treeTemplate, ...
    "NumLearningCycles",numBoostTrees, ...
    "LearnRate",learnRate);

imp = predictorImportance(finalModel);
predictors = string(finalModel.PredictorNames);
predictors = predictors(:);
imp = imp(:);

importanceTable = table(predictors, imp, ...
    'VariableNames', {'Predictor','Importance'});
importanceTable = sortrows(importanceTable, "Importance", "descend");

writetable(importanceTable, fullfile(resultsFolder, "final_feature_importance.csv"));

topN = min(40, height(importanceTable));

fig3 = figure("Visible","off");
bar(importanceTable.Importance(1:topN));
xticks(1:topN);
xticklabels(importanceTable.Predictor(1:topN));
xtickangle(60);
ylabel("Predictor Importance");
title("Top PISA Predictors - LSBoost");
grid on;
saveas(fig3, fullfile(resultsFolder, "final_feature_importance.png"));

disp("Top 30 predictors:");
disp(importanceTable(1:min(30,height(importanceTable)),:));

%% =========================
% SUMMARY REPORT
%% =========================
fileID = fopen(fullfile(resultsFolder, "final_tuned_summary_report.txt"), "w");

fprintf(fileID, "PISA 2022 Final Tuned LSBoost Wide Summary\n");
fprintf(fileID, "==========================================\n\n");
fprintf(fileID, "Target domain: %s\n", targetDomain);
fprintf(fileID, "Rows used: %d\n", height(ML));
fprintf(fileID, "Predictors used: %d\n", width(ML)-1);
fprintf(fileID, "K-folds: %d\n", kFolds);
fprintf(fileID, "Model: LSBoost Regression Ensemble\n");
fprintf(fileID, "Trees: %d\n", numBoostTrees);
fprintf(fileID, "Learn rate: %.4f\n", learnRate);
fprintf(fileID, "MaxNumSplits: %d\n", maxNumSplits);
fprintf(fileID, "MinLeafSize: %d\n\n", minLeafSize);

fprintf(fileID, "Cross-validation results:\n");
fprintf(fileID, "RMSE: %.4f +/- %.4f\n", Results.RMSE_Mean, Results.RMSE_Std);
fprintf(fileID, "MAE: %.4f +/- %.4f\n", Results.MAE_Mean, Results.MAE_Std);
fprintf(fileID, "R2: %.4f +/- %.4f\n\n", Results.R2_Mean, Results.R2_Std);

fprintf(fileID, "Interpretation:\n");
fprintf(fileID, "- This model uses many PISA questionnaire/background variables to predict mathematics performance.\n");
fprintf(fileID, "- Plausible values, IDs, weights, and technical columns are removed from predictors to reduce leakage.\n");
fprintf(fileID, "- Results should be treated as predictive associations, not causal proof.\n");
fprintf(fileID, "- Keeping country as a predictor may improve accuracy but can make country-level differences dominate interpretation.\n");

fclose(fileID);

elapsed = toc;
fprintf("\nDONE. Results saved in folder: %s\n", resultsFolder);
fprintf("Total runtime: %.2f seconds\n", elapsed);

%% ======================================================================================
% LOCAL FUNCTIONS
%% ======================================================================================

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
