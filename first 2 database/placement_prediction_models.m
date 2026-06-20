%% Board 38 - Fast Main Placement Research Pipeline
% Main research focus:
% 1) Predict placement outcome.
% 2) Identify strongest placement factors using feature importance and ablation.
%
% Secondary analysis:
% 3) Test whether salary and salary level can be predicted from the same dataset.
%
% This version is optimized to run faster than the earlier full pipeline.
%
% Required:
% Statistics and Machine Learning Toolbox
%
% Put this file in the same folder as:
% student_academic_placement_performance_dataset.csv

clear; clc; close all;
rng(42);
tic;

%% =========================
% USER SETTINGS
%% =========================
filename = "student_academic_placement_performance_dataset.csv";
resultsFolder = "board38_fast_results";

% FAST MODE:
% true  = faster for development/testing
% false = slower but stronger final run
fastMode = true;

if fastMode
    kFolds = 3;
    numTrees = 60;
    numImportanceTrees = 100;
    numAblationTrees = 40;
    runSlowModels = false;  % false skips slower ECOC/SVR/LSBoost where possible
    saveFigures = true;
else
    kFolds = 5;
    numTrees = 200;
    numImportanceTrees = 300;
    numAblationTrees = 150;
    runSlowModels = true;
    saveFigures = true;
end

% Turn these off while debugging if you want even faster runs.
runAblationStudy = true;
runRiskScores = true;
runExplanations = true;
runSalaryAnalysis = true;
runSalaryLevelAnalysis = true;

if ~exist(resultsFolder, "dir")
    mkdir(resultsFolder);
end

fprintf("Fast mode: %d\n", fastMode);
fprintf("Cross-validation folds: %d\n", kFolds);
fprintf("Bagged trees per model: %d\n", numTrees);

opts = struct();
opts.numTrees = numTrees;
opts.numImportanceTrees = numImportanceTrees;
opts.numAblationTrees = numAblationTrees;
opts.runSlowModels = runSlowModels;
opts.saveFigures = saveFigures;

%% =========================
% LOAD AND CLEAN DATA
%% =========================
Traw = readtable(filename);
Traw = clean_column_names(Traw);

% Save student IDs separately, then remove from modeling table
if ismember("student_id", string(Traw.Properties.VariableNames))
    studentIDs = Traw.student_id;
    Traw.student_id = [];
else
    studentIDs = (1:height(Traw))';
end

studentIDs = studentIDs(:);

% Convert text columns to categorical
vars = string(Traw.Properties.VariableNames);
for v = vars
    col = Traw.(v);
    if iscellstr(col) || isstring(col)
        Traw.(v) = categorical(col);
    end
end

% Remove rows with missing values and keep IDs aligned
missingRows = any(ismissing(Traw), 2);
Traw(missingRows,:) = [];
studentIDs(missingRows,:) = [];

T = Traw;

fprintf("\nRows after cleaning: %d\n", height(T));

%% =========================
% BASIC CHECKS
%% =========================
if ~ismember("placement_status", string(T.Properties.VariableNames))
    error("Missing column: placement_status");
end

if ~ismember("salary_package_lpa", string(T.Properties.VariableNames))
    error("Missing column: salary_package_lpa");
end

placedIdx = T.placement_status == 1;
avgSalary = mean(T.salary_package_lpa(placedIdx));

fprintf("Average salary among placed students: %.2f LPA\n", avgSalary);

writetable(T, fullfile(resultsFolder, "cleaned_placement_dataset.csv"));

%% ============================================================
% TASK 1: PLACEMENT STATUS PREDICTION
%% ============================================================
fprintf("\n===== TASK 1: PLACEMENT STATUS PREDICTION =====\n");

T_place = T;

% Remove salary because salary is known only after placement.
T_place.salary_package_lpa = [];

% Convert target to categorical.
T_place.placement_status = categorical(T_place.placement_status);

[placementResults, placementPred, placementTruth] = classification_cv_all_models( ...
    T_place, "placement_status", kFolds, resultsFolder, "placement_status", opts);

writetable(placementResults, fullfile(resultsFolder, "placement_status_cv_results.csv"));

%% ============================================================
% TASK 2: FEATURE IMPORTANCE
%% ============================================================
fprintf("\n===== TASK 2: FEATURE IMPORTANCE =====\n");

[importanceTable, finalPlacementModel] = placement_feature_importance( ...
    T_place, "placement_status", resultsFolder, opts);

writetable(importanceTable, fullfile(resultsFolder, "placement_feature_importance.csv"));
disp(importanceTable);

%% ============================================================
% TASK 3: ABLATION STUDY
%% ============================================================
if runAblationStudy
    fprintf("\n===== TASK 3: ABLATION STUDY =====\n");

    ablationResults = run_ablation_study(T_place, "placement_status", kFolds, resultsFolder, opts);
    writetable(ablationResults, fullfile(resultsFolder, "placement_ablation_results.csv"));
    disp(ablationResults);
else
    ablationResults = table("Skipped", NaN, NaN, ...
        'VariableNames', {'Experiment','Accuracy','MacroF1'});
end

%% ============================================================
% TASK 4: PLACEMENT RISK SCORE
%% ============================================================
if runRiskScores
    fprintf("\n===== TASK 4: PLACEMENT RISK SCORE =====\n");

    riskTable = create_placement_risk_scores(T_place, "placement_status", studentIDs, kFolds, opts);
    writetable(riskTable, fullfile(resultsFolder, "placement_risk_scores.csv"));

    disp(riskTable(1:min(10,height(riskTable)),:));
else
    riskTable = table();
end

%% ============================================================
% TASK 5: SIMPLE EXPLANATIONS
%% ============================================================
if runExplanations && ~isempty(riskTable)
    fprintf("\n===== TASK 5: SIMPLE EXPLANATIONS =====\n");

    explanationTable = create_simple_explanations( ...
        T_place, "placement_status", importanceTable, riskTable, studentIDs, 100);

    writetable(explanationTable, fullfile(resultsFolder, "placement_explanations_sample.csv"));

    disp(explanationTable(1:min(10,height(explanationTable)),:));
else
    explanationTable = table();
end

%% ============================================================
% TASK 6: SALARY PREDICTION
%% ============================================================
if runSalaryAnalysis
    fprintf("\n===== TASK 6: SALARY PREDICTION FOR PLACED STUDENTS =====\n");

    T_salary = T(T.placement_status == 1, :);

    % Remove placement_status because all remaining students are placed.
    T_salary.placement_status = [];

    [salaryResults, salaryPred, salaryTruth] = regression_cv_all_models( ...
        T_salary, "salary_package_lpa", kFolds, resultsFolder, "salary_prediction", opts);

    writetable(salaryResults, fullfile(resultsFolder, "salary_regression_cv_results.csv"));
else
    salaryResults = table("Skipped", NaN, NaN, NaN, NaN, NaN, NaN, ...
        'VariableNames', {'Model','RMSE_Mean','RMSE_Std','MAE_Mean','MAE_Std','R2_Mean','R2_Std'});
end

%% ============================================================
% TASK 7: SALARY LEVEL CLASSIFICATION AMONG PLACED STUDENTS
%% ============================================================
if runSalaryLevelAnalysis
    fprintf("\n===== TASK 7: SALARY LEVEL CLASSIFICATION AMONG PLACED STUDENTS =====\n");

    T_level = T(T.placement_status == 1, :);

    salary = T_level.salary_package_lpa;
    salaryLevel = strings(height(T_level),1);

    for i = 1:height(T_level)
        if salary(i) < avgSalary
            salaryLevel(i) = "Below_Average";
        elseif salary(i) < avgSalary * 1.20
            salaryLevel(i) = "Average";
        else
            salaryLevel(i) = "Above_Average";
        end
    end

    T_level.Salary_Level = categorical(salaryLevel);

    % Remove salary because Salary_Level was created from salary.
    T_level.salary_package_lpa = [];

    % Remove placement_status because all remaining students are placed.
    T_level.placement_status = [];

    [salaryLevelResults, salaryLevelPred, salaryLevelTruth] = classification_cv_all_models( ...
        T_level, "Salary_Level", kFolds, resultsFolder, "salary_level", opts);

    writetable(salaryLevelResults, fullfile(resultsFolder, "salary_level_cv_results.csv"));
else
    salaryLevelResults = table("Skipped", NaN, NaN, NaN, NaN, ...
        'VariableNames', {'Model','Accuracy_Mean','Accuracy_Std','MacroF1_Mean','MacroF1_Std'});
end

%% ============================================================
% FINAL SUMMARY REPORT
%% ============================================================
write_summary_report( ...
    resultsFolder, avgSalary, placementResults, salaryResults, salaryLevelResults, ablationResults, fastMode);

elapsed = toc;
fprintf("\nDONE. All results saved in folder: %s\n", resultsFolder);
fprintf("Total runtime: %.2f seconds\n", elapsed);

%% ========================================================================
% LOCAL FUNCTIONS
%% ========================================================================

function T = clean_column_names(T)
    oldNames = string(T.Properties.VariableNames);
    newNames = lower(strtrim(oldNames));
    newNames = replace(newNames, " ", "_");
    newNames = replace(newNames, "-", "_");
    newNames = matlab.lang.makeValidName(newNames);
    T.Properties.VariableNames = cellstr(newNames);
end

function [Results, bestPred, yAll] = classification_cv_all_models(T, target, k, resultsFolder, prefix, opts)

    T = rmmissing(T);

    yAll = T.(target);
    if ~iscategorical(yAll)
        yAll = categorical(yAll);
        T.(target) = yAll;
    end

    nClasses = numel(categories(yAll));

    if nClasses == 2
        modelNames = ["Decision Tree", "Bagged Trees"];
        if opts.runSlowModels
            modelNames = ["Decision Tree", "Bagged Trees", "ECOC"];
        end
    else
        modelNames = ["Decision Tree", "Bagged Trees"];
        if opts.runSlowModels
            modelNames = ["Decision Tree", "Bagged Trees", "ECOC"];
        end
    end

    numModels = numel(modelNames);

    cv = cvpartition(height(T), "KFold", k);

    accAll = zeros(k,numModels);
    f1All = zeros(k,numModels);
    predOOF = strings(height(T),numModels);

    for fold = 1:k

        trainIdx = training(cv,fold);
        testIdx = test(cv,fold);

        trainT = T(trainIdx,:);
        testT = T(testIdx,:);

        yTrain = trainT.(target);
        yTest = testT.(target);

        XTrain = removevars(trainT,target);
        XTest = removevars(testT,target);

        for m = 1:numModels

            name = modelNames(m);

            switch name
                case "Decision Tree"
                    mdl = fitctree(XTrain,yTrain);

                case "Bagged Trees"
                    mdl = fitcensemble(XTrain,yTrain, ...
                        "Method","Bag", ...
                        "NumLearningCycles",opts.numTrees);

                case "ECOC"
                    mdl = fitcecoc(XTrain,yTrain);

                otherwise
                    error("Unknown classification model: %s", name);
            end

            yPred = predict(mdl,XTest);

            accAll(fold,m) = mean(yPred == yTest);
            f1All(fold,m) = macro_f1(yTest,yPred);
            predOOF(testIdx,m) = string(yPred);

        end

        fprintf("Fold %d complete.\n", fold);
    end

    Results = table;
    Results.Model = modelNames';
    Results.Accuracy_Mean = mean(accAll,1)';
    Results.Accuracy_Std = std(accAll,0,1)';
    Results.MacroF1_Mean = mean(f1All,1)';
    Results.MacroF1_Std = std(f1All,0,1)';

    Results = sortrows(Results,"Accuracy_Mean","descend");
    disp(Results);

    bestModelName = Results.Model(1);
    bestIdx = find(modelNames == bestModelName,1);
    bestPred = categorical(predOOF(:,bestIdx), categories(yAll));

    if opts.saveFigures
        fig = figure("Visible","off");
        confusionchart(yAll,bestPred);
        title(strrep(prefix,"_"," ") + " - CV Confusion Matrix (" + bestModelName + ")");
        saveas(fig, fullfile(resultsFolder, prefix + "_confusion_matrix.png"));
        close(fig);
    end

end

function [Results, bestPred, yAll] = regression_cv_all_models(T, target, k, resultsFolder, prefix, opts)

    T = rmmissing(T);
    yAll = T.(target);

    if opts.runSlowModels
        modelNames = ["Linear Regression", ...
                      "Regression Tree", ...
                      "Bagged Trees", ...
                      "LSBoost", ...
                      "Support Vector Regression"];
    else
        modelNames = ["Linear Regression", ...
                      "Regression Tree", ...
                      "Bagged Trees"];
    end

    numModels = numel(modelNames);
    cv = cvpartition(height(T), "KFold", k);

    rmseAll = zeros(k,numModels);
    maeAll = zeros(k,numModels);
    r2All = zeros(k,numModels);
    predOOF = zeros(height(T),numModels);

    for fold = 1:k

        trainIdx = training(cv,fold);
        testIdx = test(cv,fold);

        trainT = T(trainIdx,:);
        testT = T(testIdx,:);

        yTrain = trainT.(target);
        yTest = testT.(target);

        XTrain = removevars(trainT,target);
        XTest = removevars(testT,target);

        for m = 1:numModels

            name = modelNames(m);

            switch name
                case "Linear Regression"
                    mdl = fitrlinear(XTrain,yTrain);

                case "Regression Tree"
                    mdl = fitrtree(XTrain,yTrain);

                case "Bagged Trees"
                    mdl = fitrensemble(XTrain,yTrain, ...
                        "Method","Bag", ...
                        "NumLearningCycles",opts.numTrees);

                case "LSBoost"
                    mdl = fitrensemble(XTrain,yTrain, ...
                        "Method","LSBoost", ...
                        "NumLearningCycles",opts.numTrees);

                case "Support Vector Regression"
                    mdl = fitrsvm(XTrain,yTrain, ...
                        "KernelFunction","gaussian", ...
                        "Standardize",true);

                otherwise
                    error("Unknown regression model: %s", name);
            end

            yPred = predict(mdl,XTest);
            predOOF(testIdx,m) = yPred;

            rmseAll(fold,m) = sqrt(mean((yPred-yTest).^2));
            maeAll(fold,m) = mean(abs(yPred-yTest));
            r2All(fold,m) = 1 - sum((yPred-yTest).^2) / sum((yTest-mean(yTest)).^2);

        end

        fprintf("Fold %d complete.\n", fold);
    end

    Results = table;
    Results.Model = modelNames';
    Results.RMSE_Mean = mean(rmseAll,1)';
    Results.RMSE_Std = std(rmseAll,0,1)';
    Results.MAE_Mean = mean(maeAll,1)';
    Results.MAE_Std = std(maeAll,0,1)';
    Results.R2_Mean = mean(r2All,1)';
    Results.R2_Std = std(r2All,0,1)';

    Results = sortrows(Results,"RMSE_Mean");
    disp(Results);

    bestModelName = Results.Model(1);
    bestIdx = find(modelNames == bestModelName,1);
    bestPred = predOOF(:,bestIdx);

    if opts.saveFigures
        fig1 = figure("Visible","off");
        scatter(yAll,bestPred,18,"filled");
        hold on;

        minVal = min([yAll; bestPred]);
        maxVal = max([yAll; bestPred]);

        plot([minVal maxVal],[minVal maxVal],"r--","LineWidth",1.5);
        grid on;
        xlabel("Actual");
        ylabel("Predicted");
        title(strrep(prefix,"_"," ") + " - Predicted vs Actual (" + bestModelName + ")");
        hold off;

        saveas(fig1, fullfile(resultsFolder, prefix + "_predicted_vs_actual.png"));
        close(fig1);

        residuals = yAll - bestPred;

        fig2 = figure("Visible","off");
        scatter(bestPred,residuals,18,"filled");
        hold on;
        yline(0,"r--","LineWidth",1.5);
        grid on;
        xlabel("Predicted");
        ylabel("Residual");
        title(strrep(prefix,"_"," ") + " - Residual Plot (" + bestModelName + ")");
        hold off;

        saveas(fig2, fullfile(resultsFolder, prefix + "_residuals.png"));
        close(fig2);
    end

end

function [importanceTable, mdl] = placement_feature_importance(T, target, resultsFolder, opts)

    y = T.(target);

    if ~iscategorical(y)
        y = categorical(y);
        T.(target) = y;
    end

    X = removevars(T,target);

    mdl = fitcensemble(X,y, ...
        "Method","Bag", ...
        "NumLearningCycles",opts.numImportanceTrees);

    imp = predictorImportance(mdl);

    predictors = string(mdl.PredictorNames);
    predictors = predictors(:);
    imp = imp(:);

    if numel(predictors) ~= numel(imp)
        error("Predictor names and importance values do not have the same length.");
    end

    importanceTable = table(predictors, imp, ...
        'VariableNames', {'Predictor','Importance'});

    importanceTable = sortrows(importanceTable,"Importance","descend");

    if opts.saveFigures
        fig = figure("Visible","off");
        bar(importanceTable.Importance);
        xticks(1:height(importanceTable));
        xticklabels(importanceTable.Predictor);
        xtickangle(45);
        ylabel("Predictor Importance");
        title("Placement Feature Importance");
        grid on;

        saveas(fig, fullfile(resultsFolder, "placement_feature_importance.png"));
        close(fig);
    end

end

function Results = run_ablation_study(T, target, k, resultsFolder, opts)

    groups = {
        "All Features", strings(1,0);
        "Remove Academic Scores", ["ssc_percentage","hsc_percentage","degree_percentage","cgpa","entrance_exam_score"];
        "Remove Skill Scores", ["technical_skill_score","soft_skill_score"];
        "Remove Experience", ["internship_count","live_projects","work_experience_months","certifications"];
        "Remove Attendance/Backlogs", ["attendance_percentage","backlogs"];
        "Remove Demographic/Activities", ["gender","extracurricular_activities"]
    };

    nGroups = size(groups,1);

    names = strings(nGroups,1);
    acc = zeros(nGroups,1);
    f1 = zeros(nGroups,1);

    for i = 1:nGroups

        names(i) = groups{i,1};
        varsToRemove = groups{i,2};

        Ttmp = T;

        for v = varsToRemove
            if ismember(v, string(Ttmp.Properties.VariableNames))
                Ttmp.(v) = [];
            end
        end

        [acc(i), f1(i)] = quick_bagged_classification_cv(Ttmp,target,k,opts);

        fprintf("%s | Accuracy %.4f | MacroF1 %.4f\n", names(i), acc(i), f1(i));

    end

    Results = table(names,acc,f1, ...
        'VariableNames', {'Experiment','Accuracy','MacroF1'});

    Results = sortrows(Results,"Accuracy","descend");

    if opts.saveFigures
        fig = figure("Visible","off");
        bar(Results.Accuracy);
        xticks(1:height(Results));
        xticklabels(Results.Experiment);
        xtickangle(45);
        ylabel("CV Accuracy");
        title("Ablation Study: Placement Prediction");
        grid on;

        saveas(fig, fullfile(resultsFolder, "placement_ablation_accuracy.png"));
        close(fig);
    end

end

function [accMean, f1Mean] = quick_bagged_classification_cv(T, target, k, opts)

    y = T.(target);

    if ~iscategorical(y)
        y = categorical(y);
        T.(target) = y;
    end

    cv = cvpartition(height(T),"KFold",k);

    acc = zeros(k,1);
    f1 = zeros(k,1);

    for fold = 1:k

        trainT = T(training(cv,fold),:);
        testT = T(test(cv,fold),:);

        XTrain = removevars(trainT,target);
        yTrain = trainT.(target);

        XTest = removevars(testT,target);
        yTest = testT.(target);

        mdl = fitcensemble(XTrain,yTrain, ...
            "Method","Bag", ...
            "NumLearningCycles",opts.numAblationTrees);

        yPred = predict(mdl,XTest);

        acc(fold) = mean(yPred == yTest);
        f1(fold) = macro_f1(yTest,yPred);

    end

    accMean = mean(acc);
    f1Mean = mean(f1);

end

function riskTable = create_placement_risk_scores(T, target, studentIDs, k, opts)

    y = T.(target);

    if ~iscategorical(y)
        y = categorical(y);
        T.(target) = y;
    end

    cv = cvpartition(height(T),"KFold",k);

    pPlaced = zeros(height(T),1);

    for fold = 1:k

        trainT = T(training(cv,fold),:);
        testT = T(test(cv,fold),:);

        XTrain = removevars(trainT,target);
        yTrain = trainT.(target);

        XTest = removevars(testT,target);

        mdl = fitcensemble(XTrain,yTrain, ...
            "Method","Bag", ...
            "NumLearningCycles",opts.numTrees);

        [~,score] = predict(mdl,XTest);

        classNames = string(mdl.ClassNames);
        idxPlaced = find(classNames == "1",1);

        if isempty(idxPlaced)
            idxPlaced = numel(classNames);
        end

        pPlaced(test(cv,fold)) = score(:,idxPlaced);

    end

    riskScore = round(100 * (1 - pPlaced),2);

    riskGroup = strings(height(T),1);
    riskGroup(riskScore < 30) = "Low_Risk";
    riskGroup(riskScore >= 30 & riskScore < 70) = "Medium_Risk";
    riskGroup(riskScore >= 70) = "High_Risk";

    riskTable = table(studentIDs(1:height(T)), string(T.(target)), round(pPlaced,4), riskScore, categorical(riskGroup), ...
        'VariableNames', {'Student_ID','Actual_Placement','Predicted_Probability_Placed','Placement_Risk_Score','Risk_Group'});

end

function explanationTable = create_simple_explanations(T, target, importanceTable, riskTable, studentIDs, maxRows)

    y = T.(target);

    if ~iscategorical(y)
        y = categorical(y);
    end

    predictors = importanceTable.Predictor;
    numericPredictors = strings(0,1);

    for i = 1:numel(predictors)
        p = predictors(i);

        if ismember(p, string(T.Properties.VariableNames)) && isnumeric(T.(p))
            numericPredictors(end+1,1) = p; %#ok<AGROW>
        end
    end

    n = min(maxRows,height(T));

    topStrengths = strings(n,1);
    topRisks = strings(n,1);

    placedMask = string(y) == "1";

    for row = 1:n

        strengths = strings(0,1);
        risks = strings(0,1);

        for j = 1:numel(numericPredictors)

            p = numericPredictors(j);
            x = T.(p);

            meanPlaced = mean(x(placedMask));
            meanNotPlaced = mean(x(~placedMask));

            direction = meanPlaced - meanNotPlaced;

            lowQ = quantile(x,0.33);
            highQ = quantile(x,0.66);

            val = x(row);

            if direction >= 0
                if val >= highQ
                    strengths(end+1,1) = p + " high"; %#ok<AGROW>
                elseif val <= lowQ
                    risks(end+1,1) = p + " low"; %#ok<AGROW>
                end
            else
                if val >= highQ
                    risks(end+1,1) = p + " high"; %#ok<AGROW>
                elseif val <= lowQ
                    strengths(end+1,1) = p + " low"; %#ok<AGROW>
                end
            end

        end

        if isempty(strengths)
            topStrengths(row) = "No major strength flagged";
        else
            topStrengths(row) = strjoin(strengths(1:min(3,numel(strengths))), "; ");
        end

        if isempty(risks)
            topRisks(row) = "No major risk flagged";
        else
            topRisks(row) = strjoin(risks(1:min(3,numel(risks))), "; ");
        end

    end

    explanationTable = table(studentIDs(1:n), riskTable.Placement_Risk_Score(1:n), riskTable.Risk_Group(1:n), ...
        topStrengths, topRisks, ...
        'VariableNames', {'Student_ID','Placement_Risk_Score','Risk_Group','Top_Strengths','Top_Risk_Factors'});

end

function f1 = macro_f1(yTrue,yPred)

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

        precision = tp / max(tp+fp,1);
        recall = tp / max(tp+fn,1);

        f1Scores(i) = 2 * precision * recall / max(precision+recall,1e-12);

    end

    f1 = mean(f1Scores);

end

function write_summary_report(resultsFolder, avgSalary, placementResults, salaryResults, salaryLevelResults, ablationResults, fastMode)

    fileID = fopen(fullfile(resultsFolder, "board38_fast_summary_report.txt"), "w");

    fprintf(fileID, "Board 38 Fast Placement Research Summary\n");
    fprintf(fileID, "=======================================\n\n");

    fprintf(fileID, "Fast mode: %d\n", fastMode);
    fprintf(fileID, "Average placed-student salary: %.2f LPA\n\n", avgSalary);

    fprintf(fileID, "1) Placement prediction\n");
    fprintf(fileID, "Best model: %s\n", placementResults.Model(1));
    fprintf(fileID, "Accuracy: %.4f\n", placementResults.Accuracy_Mean(1));
    fprintf(fileID, "MacroF1: %.4f\n\n", placementResults.MacroF1_Mean(1));

    fprintf(fileID, "2) Salary prediction for placed students\n");
    fprintf(fileID, "Best model by RMSE: %s\n", salaryResults.Model(1));
    fprintf(fileID, "RMSE: %.4f\n", salaryResults.RMSE_Mean(1));
    fprintf(fileID, "MAE: %.4f\n", salaryResults.MAE_Mean(1));
    fprintf(fileID, "R2: %.4f\n\n", salaryResults.R2_Mean(1));

    fprintf(fileID, "3) Salary-level classification among placed students\n");
    fprintf(fileID, "Best model: %s\n", salaryLevelResults.Model(1));
    fprintf(fileID, "Accuracy: %.4f\n", salaryLevelResults.Accuracy_Mean(1));
    fprintf(fileID, "MacroF1: %.4f\n\n", salaryLevelResults.MacroF1_Mean(1));

    fprintf(fileID, "4) Ablation study\n");
    fprintf(fileID, "Best ablation setting: %s\n", ablationResults.Experiment(1));
    fprintf(fileID, "Accuracy: %.4f\n", ablationResults.Accuracy(1));
    fprintf(fileID, "MacroF1: %.4f\n\n", ablationResults.MacroF1(1));

    fprintf(fileID, "Main interpretation:\n");
    fprintf(fileID, "- This dataset is strong for predicting placement status.\n");
    fprintf(fileID, "- Feature importance and ablation are the strongest parts of the research.\n");
    fprintf(fileID, "- Salary prediction is weak because the dataset lacks job-market variables such as role, company, industry, location, and interview score.\n");
    fprintf(fileID, "- Salary-level classification is secondary and should not be presented as exact job prediction.\n");
    fprintf(fileID, "- Placement risk scores are useful for explanation, but they should not be treated as causal or final decisions.\n");

    fclose(fileID);

end
