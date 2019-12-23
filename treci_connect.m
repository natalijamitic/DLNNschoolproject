clc, clear, close all

t = readtable('connect-4.csv');
t = table2array(t);
sz = size(t);
ulazi = t(:,1:sz(1,2)-1);
strIzlazi = t(:,sz(1,2));
CLASS_NUM = 3;

ulazi = cellfun(@double, ulazi);
izlazi = [];
for i = 1:sz(1,1)
    k = strIzlazi(i,1);
    k = k{1};
    k = k(1);
    if k == 'w';
        izlazi = [ izlazi; 1 ];
    elseif k == 'l';
        izlazi = [ izlazi; 2 ];
    else
        izlazi = [ izlazi; 3 ];
    end
end

szz = size(ulazi)

for i = 1:szz(1,1)
    for j = 1:szz(1,2)
        if ulazi(i,j) == 98
            ulazi(i,j) = 0;
        elseif ulazi(i,j) == 120
            ulazi(i,j) = 1;
        else ulazi(i,j) = -1;
        end
    end
end


ulazi = rot90(ulazi);
izlazi = rot90(izlazi);

K = cell(1,CLASS_NUM);
for i = 1:CLASS_NUM
    K{1,i} = ulazi(:,izlazi == i);
    hold all
    sz = size(K{1,i});
    n = sz(1,2);
    scatter(i,n,'.');
end

trainUlaz = [];
testUlaz = [];
trainIzlaz = [];
testIzlaz = [];

for i = 1:CLASS_NUM
    sz = size(K{1,i});
    n = sz(1,2);
    granica = ceil(0.8*n);
    trainUlaz = [ trainUlaz [K{1,i}(:, 1:granica)]];
    testUlaz = [ testUlaz [K{1,i}(:, granica+1:n)]];
    trainIzlaz = [ trainIzlaz  i*ones(1,granica) ];
    testIzlaz = [ testIzlaz i*ones(1,n-granica) ];
end

ind = randperm(length(trainUlaz));
trainUlaz = trainUlaz(:,ind);
trainIzlaz = trainIzlaz(:,ind);

ind = randperm(length(testUlaz));
testUlaz = testUlaz(:,ind);
testIzlaz = testIzlaz(:,ind);

% trainMutantUlaz = [];
% trainMutantIzlaz = [];
% Nmutant = 5159;
% for i = 1:CLASS_NUM
%     trainMutantUlaz = [ trainMutantUlaz [K{1,i}(:, 1:Nmutant)]];
%     trainMutantIzlaz = [ trainMutantIzlaz  i*ones(1,Nmutant) ];
% end
% ind = randperm(length(trainMutantUlaz));
% trainMutantUlaz = trainMutantUlaz(:,ind);
% trainMutantIzlaz = trainMutantIzlaz(:,ind);

Ntrain = length(trainUlaz);

acc = 0;
bestStruct = [32 32 64 64 128 256 16];
bestActivation = 'logsig';
bestRegularization = 0.14;
bestEpochNum = 150;


net = patternnet(bestStruct);
net.layers{1}.transferFcn = 'poslin';
net.layers{2}.transferFcn = 'poslin';
net.layers{3}.transferFcn = 'poslin';
net.layers{4}.transferFcn = 'poslin';
net.layers{5}.transferFcn = 'poslin';
net.layers{6}.transferFcn = 'poslin';
net.layers{7}.transferFcn = 'softmax';


net.performParam.regularization = bestRegularization;
% net.divideFcn = 'divideint';
net.divideFcn = '';
net.trainParam.max_fail = 10;
net.trainParam.goal = 10e-9;
net.trainParam.min_grad = 10e-9;
net.trainParam.epochs = bestEpochNum;
net.trainparam.showWindow = true;

w = ones(Ntrain, 1);
for i = 1:CLASS_NUM
    sz = size(K{1,i});
    w(trainIzlaz == i) = length(trainIzlaz)/(CLASS_NUM*sz(1,2));
end

net = train(net, trainUlaz, trainIzlaz, [], [], w);

Ntest = length(testUlaz);

%PREDIKCIJA NAD TRAIN SKUPOM
Ytrain_pred = net(trainUlaz);
targets = zeros(CLASS_NUM, length(trainIzlaz));
outputs = zeros(CLASS_NUM, length(Ytrain_pred));
targetsIdx = sub2ind(size(targets), trainIzlaz, 1:length(trainIzlaz));
outputsIdx = sub2ind(size(outputs), Ytrain_pred, 1:length(Ytrain_pred));
outputsIdx = arrayfun(@round, outputsIdx);
targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;
figure
plotconfusion(targets, outputs);

[c, cm] = confusion(targets, outputs);
cm = cm'

%PREDIKCIJA NAD TEST SKUPOM
Ytest_pred = net(testUlaz);
targets = zeros(CLASS_NUM, length(testIzlaz));
outputs = zeros(CLASS_NUM, length(Ytest_pred));
targetsIdx = sub2ind(size(targets), testIzlaz, 1:length(testIzlaz));
outputsIdx = sub2ind(size(outputs), Ytest_pred, 1:length(Ytest_pred));
outputsIdx = arrayfun(@round, outputsIdx);
targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;
figure
plotconfusion(targets, outputs);

[c, cm] = confusion(targets, outputs);
cm = cm'

 R = cm(1, 1) / (cm(1, 1) + cm(2, 1))
 P = cm(1, 1) / (cm(1, 1) + cm(1, 2))
 A = (cm(1, 1) + cm(2, 2)) / (cm(1, 1) + cm(1, 2) + cm(2, 1) + cm(2, 2))
 F1 = 2 * P * R / (P + R)