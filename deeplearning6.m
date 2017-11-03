function [weights, biases, acc] = deeplearning6(trainimg,trainlabel,testimg,testlabel)
%% Data Processing 
trainimg = trainimg';
testimg = testimg';
ntest = numel(testlabel);   %number of test data
n0 = numel(trainlabel); %number of training data
f0 = numel(trainimg)/n0;    %number of features
testlabel_ph = zeros(ntest,10);
testlabel_ph(sub2ind([ntest 10],(1:ntest)',testlabel+1))=1;
%% Initializing Network
%% Convolutional Layers
% Filter X, Filter Y, Filter Z, Stride, Output Channels
LP = [5 5 1   1 28 28   4 ;
      4 4 4   2 14 14   8 ;
      4 4 8   2  7  7  12 ; 
      7 7 12  0  1  1 200 ;
      1 1 200 0  1  1  10];
pad = (LP(:,5:6)-1).*repmat(LP(:,4),1,2)+LP(:,1:2)-[28 28; LP(1:end-1,5:6)];
nlayers = size(LP,1);
network = [f0 LP(:,end)'];
for i = 1:size(LP,1)
    W{i} = rand(network(i),network(i+1))-0.5;
    B{i} = rand(1,network(i+1))-0.5;
end
%% Others
iter = 0; xiter=[]; frame = [];
k = 100;
trainA_p=[];testA_p=[]; XE_p =[]; XEtest_p=[]; 
%% Training
    xiter = [xiter iter];
    k1= rem(iter,ceil(n0/k));
    imgset = trainimg(1+k1*k:min(n0,(k1+1)*k),:);
    labelset = trainlabel(1+k1*k:min(n0,(k1+1)*k));
    labelset_ph = zeros(k,nout);
    labelset_ph(sub2ind([k nout],(1:k)',labelset+1))=1;
    
    X{1} = 
end