function [W, B, acc] = deeplearning6()
%%  Data Processing and Labeling

% img = img';
% testratio = 0.1; % percentage test data
% testimg = img(1:1/testratio:numel(label),:);
% testlabel = label(1:1/testratio:numel(label),:);
% ntest = numel(testlabel);   %number of test data
% testlabel_ph = zeros(ntest,10);
% testlabel_ph(sub2ind([ntest 10],(1:ntest)',testlabel+1))=1;
% img(1:1/testratio:numel(label),:)=[];
% label(1:1/testratio:numel(label))=[];
% trainimg = img;
% trainlabel = label;
% n0 = numel(trainlabel); %number of training data
% f0 = numel(trainimg)/n0;    %number of features
trainimg = loadMNISTImages('train-images.idx3-ubyte');
trainlabel = loadMNISTLabels('train-labels.idx1-ubyte');
testimg = loadMNISTImages('t10k-images.idx3-ubyte');
testlabel = loadMNISTLabels('t10k-labels.idx1-ubyte');

trainimg = trainimg';
testimg = testimg';
ntest = numel(testlabel);   %number of test data
n0 = numel(trainlabel); %number of training data
f0 = numel(trainimg)/n0;    %number of features
testlabel_ph = zeros(ntest,10);
testlabel_ph(sub2ind([ntest 10],(1:ntest)',testlabel+1))=1;

%% Plots Initialization

%%  Initializing Network

network = [200 100 60 30 10];
network = [f0 200 100 60 30 10];
nlayer = numel(network)-1;
nout = network(end);
reg = 0.1;

%Layers
for i=1:nlayer
    W{i} = rand(network(i),network(i+1))-0.5;
    B{i} = rand(1,network(i+1))-0.5;
end

%% Others
iter = 0; xiter=[]; frame = [];
k = 100;
trainA_p=[];testA_p=[]; XE_p =[]; XEtest_p=[]; N1_p =[]; N2_p = [];
%%  Training Network
while iter<10000
    xiter = [xiter iter];
    
    k1= rem(iter,ceil(n0/k));
    imgset = trainimg(1+k1*k:min(n0,(k1+1)*k),:);
    labelset = trainlabel(1+k1*k:min(n0,(k1+1)*k));
    labelset_ph = zeros(k,nout);
    labelset_ph(sub2ind([k nout],(1:k)',labelset+1))=1;
    
    X{1} = imgset*W{1}+repmat(B{1},k,1); %weighted sum + bias
    Y{1} = (1+exp(-X{1})).^(-1);    %sigmoid
    %Y{1} = max(0,X{1}); %relu
    for i=2:nlayer-1
        X{i} = Y{i-1}*W{i}+repmat(B{i},k,1); %weighted sum + bias
        Y{i} = (1+exp(-X{i})).^(-1);    %sigmoid
    end
    if nlayer>1
        X{nlayer} = Y{nlayer-1}*W{nlayer}+repmat(B{nlayer},k,1);
        Y{nlayer} = exp(X{nlayer})./repmat(sum(exp(X{nlayer}),2),1,nout); %softmax
    else
        X{nlayer} = imgset*W{nlayer}+repmat(B{nlayer},k,1);
        Y{nlayer} = exp(X{nlayer})./repmat(sum(exp(X{nlayer}),2),1,nout); %softmax
    end
    
    %% Cost Function
    XE = -sum(sum(log(Y{nlayer}).*labelset_ph))/k;  %Cross Entropy
    XE_p = [XE_p XE];
    
    %% Optimizer
    img_ph = zeros(k,network(end));
    img_ph(sub2ind(size(img_ph),(1:k)',labelset+1)) = 1;
    
    if nlayer==1
        dXEdW{nlayer} = permute(repmat(imgset,1,1,nout),[1 3 2]).*repmat(Y{end}-img_ph,1,1,f0);
        dXEdW{nlayer} = permute(sum(dXEdW{nlayer})/k,[3 2 1]) + W{nlayer}*reg/k;
        dXEdB{nlayer} = sum(Y{end}-img_ph)/k;
    else
        dXEdW{nlayer} = permute(repmat(Y{end-1},1,1,nout),[1 3 2]).*repmat(Y{end}-img_ph,1,1,network(end-1));
        dXEdW{nlayer} = permute(sum(dXEdW{nlayer}),[3 2 1]);
        dXEdB{nlayer} = sum(Y{end}-img_ph)/k;
        
        %sumel = repmat(sum(exp(X{end}),2),1,network(end));
        %err{nlayer} = (-img_ph./(Y{end})).*(sumel.*exp(X{end})-exp(X{end}).^2)./sumel.^2;%(sum*eL - eL^2)/sum^2
        err{nlayer} = Y{end} - img_ph;
        
        for i=nlayer-1:-1:2
            err{i} = (err{i+1}*(W{i+1})').*(Y{i}).*(1-Y{i});
            dXEdW{i} = (Y{i-1})'*err{i}/k;
            dXEdB{i} = sum(err{i})/k;
        end    
        end
        
        err{1} = (err{2}*W{2}').*Y{1}.*(1-Y{1});
        dXEdW{1}=imgset'*err{1}/k;
        dXEdB{1}=sum(err{1})/k;
    
    %% Numerical Gradient Checking
    
%     eps = zeros(f0,network(2)); XEp1 = zeros(f0,network(2));XEp2 = XEp1;
%     for i = 1:network(2)
%         for j = 1:f0
%         eps = zeros(f0,network(2));
%         eps(j,i) = 10^-4;
%         
%     Wp1{1} = W{1}+eps;
%     Wp2{1} = W{1}-eps;
%     
%     Xp1{1} = imgset*Wp1{1}+repmat(B{1},k,1); %weighted sum + bias
%     Yp1{1} = (1+exp(-Xp1{1})).^(-1);    %sigmoid
%     for i=2:nlayer-1
%         Xp1{i} = Yp1{i-1}*W{i}+repmat(B{i},k,1); %weighted sum + bias
%         Yp1{i} = (1+exp(-Xp1{i})).^(-1);    %sigmoid
%     end
%         Xp1{nlayer} = Yp1{nlayer-1}*W{nlayer}+repmat(B{nlayer},k,1);
%         Yp1{nlayer} = exp(Xp1{nlayer})./repmat(sum(exp(Xp1{nlayer}),2),1,nout); %softmax
%     XEp1(j,i) = -sum(sum(log(Yp1{nlayer}).*labelset_ph))/k;  %Cross Entropy
% 
%     Xp2{1} = imgset*Wp2{1}+repmat(B{1},k,1); %weighted sum + bias
%     Yp2{1} = (1+exp(-Xp2{1})).^(-1);    %sigmoid
%     for i=2:nlayer-1
%         Xp2{i} = Yp2{i-1}*W{i}+repmat(B{i},k,1); %weighted sum + bias
%         Yp2{i} = (1+exp(-Xp2{i})).^(-1);    %sigmoid
%     end
%         Xp2{nlayer} = Yp2{nlayer-1}*W{nlayer}+repmat(B{nlayer},k,1);
%         Yp2{nlayer} = exp(Xp2{nlayer})./repmat(sum(exp(Xp2{nlayer}),2),1,nout); %softmax
%     XEp2(j,i) = -sum(sum(log(Yp2{nlayer}).*labelset_ph))/k;  %Cross Entropy
%         end
%     end
%     XEp = (XEp1 - XEp2)/(2*10^-4);
%     grdchk1 = sum(sum((XEp - dXEdW{1}).^2))/numel(XEp)
% 
%     %% Layer 2
%     XEp1 = zeros(network(2),network(3));XEp2 = XEp1;
%     for i = 1:network(3)
%         for j = 1:network(2)
%         eps = zeros(network(2),network(3));
%         eps(j,i) = 10^-4;
%         
%     Wp1{2} = W{2}+eps;
%     Wp2{2} = W{2}-eps;
%     
%     Xp1{1} = imgset*W{1}+repmat(B{1},k,1); %weighted sum + bias
%     Yp1{1} = (1+exp(-Xp1{1})).^(-1);    %sigmoid
%     for i=2:nlayer-1
%         Xp1{i} = Yp1{i-1}*W{i}+repmat(B{i},k,1); %weighted sum + bias
%         Yp1{i} = (1+exp(-Xp1{i})).^(-1);    %sigmoid
%     end
%         Xp1{nlayer} = Yp1{nlayer-1}*Wp1{nlayer}+repmat(B{nlayer},k,1);
%         Yp1{nlayer} = exp(Xp1{nlayer})./repmat(sum(exp(Xp1{nlayer}),2),1,nout); %softmax
%     XEp1(j,i) = -sum(sum(log(Yp1{nlayer}).*labelset_ph))/k;  %Cross Entropy
% 
%     Xp2{1} = imgset*W{1}+repmat(B{1},k,1); %weighted sum + bias
%     Yp2{1} = (1+exp(-Xp2{1})).^(-1);    %sigmoid
%     for i=2:nlayer-1
%         Xp2{i} = Yp2{i-1}*W{i}+repmat(B{i},k,1); %weighted sum + bias
%         Yp2{i} = (1+exp(-Xp2{i})).^(-1);    %sigmoid
%     end
%         Xp2{nlayer} = Yp2{nlayer-1}*Wp2{nlayer}+repmat(B{nlayer},k,1);
%         Yp2{nlayer} = exp(Xp2{nlayer})./repmat(sum(exp(Xp2{nlayer}),2),1,nout); %softmax
%     XEp2(j,i) = -sum(sum(log(Yp2{nlayer}).*labelset_ph))/k;  %Cross Entropy
%         end
%     end
%     XEp = (XEp1 - XEp2)/(2*10^-4);
%     grdchk2 = sum(sum((XEp - dXEdW{2}).^2))/numel(XEp)
    
    
    %% Growth
    gr = k*(0.0029*exp(-0.01*iter)+0.0001); %growth function
    %gr = 0.003*k;
    
    
    for i = 1:nlayer
        W{i} = W{i} - gr.*dXEdW{i};
        B{i} = B{i} - gr.*dXEdB{i};
    end
    
    %% Regularization
    
    %% Accuracy
    [~,idx] = max(Y{nlayer},[],2);
    trainA = numel(find(idx==labelset+1))/k;    %training accuracy
    trainA_p = [trainA_p trainA];
    
    iter = iter + 1;
    
    %%  Testing Model
    X{1} = testimg*W{1}+repmat(B{1},ntest,1); %weighted sum + bias
    Y{1} = (1+exp(-X{1})).^(-1);    %sigmoid
    for i=2:nlayer-1
        X{i} = Y{i-1}*W{i}+repmat(B{i},ntest,1); %weighted sum + bias
        Y{i} = (1+exp(-X{i})).^(-1);    %sigmoid
    end
    if nlayer>1
        X{nlayer} = Y{nlayer-1}*W{nlayer}+repmat(B{nlayer},ntest,1);
        Y{nlayer} = exp(X{nlayer})./repmat(sum(exp(X{nlayer}),2),1,nout); %softmax
    else
        X{nlayer} = testimg*W{nlayer}+repmat(B{nlayer},ntest,1);
        Y{nlayer} = exp(X{nlayer})./repmat(sum(exp(X{nlayer}),2),1,nout); %softmax
    end
    
    %%% Cost Function
    XEtest = -sum(sum(log(Y{nlayer}).*testlabel_ph))/ntest;  %Cross Entropy
    XEtest_p = [XEtest_p XEtest];
    
    %%% Accuracy
    [~,testidx] = max(Y{nlayer},[],2);
    testA = numel(find(testidx==testlabel+1))/ntest;    %training accuracy
    testA_p = [testA_p testA];
    acc=max(testA_p)
    %% Plots
%     subplot(2,2,1)
%     plot(xiter,trainA_p,'Color',[0 0 1])
%     ylim([0 1])
%     hold on
%     plot(xiter,testA_p,'Color',[1 0 0])
%     title('Accuracy');
%     
%     subplot(2,2,3)
%     plot(xiter,XE_p)
%     ylim([0 4])
%     hold on
%     plot(xiter,XEtest_p)
%     title('Cross Entropy')
%     hold off
%     
%     subplot(2,2,2)
%     N1_ph = reshape(W{1},1,numel(W{1}));
%     N1_p = [N1_p [max(N1_ph) ; min(N1_ph) ; mean(N1_ph); mean(N1_ph)+std(N1_ph) ; mean(N1_ph)-std(N1_ph)]];
%     plot(xiter,N1_p)
%     title('Weights');
%     
%     if nlayer>1
%     subplot(2,2,4)
%     N2_ph = reshape(W{2},1,numel(W{2}));
%     N2_p = [N2_p [max(N2_ph) ; min(N2_ph) ; mean(N2_ph); mean(N2_ph)+std(N2_ph) ; mean(N2_ph)-std(N2_ph)]];
%     plot(xiter,N2_p)
%     title('Weights');
%     end
%     
%     frame = [frame getframe];
end