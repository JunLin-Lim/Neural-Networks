function [N1, B1] = deeplearning4(img,label)

img = img';
testratio = 0.1; % percentage test data
testimg = img(1:1/testratio:numel(label),:);
testlabel = label(1:1/testratio:numel(label),:);
ntest = numel(testlabel);
img(1:1/testratio:numel(label),:)=[];
label(1:1/testratio:numel(label))=[];
trainimg = img;
trainlabel = label;

n0 = numel(trainlabel);
f0 = numel(trainimg)/n0;
iter = 0; epoch = 0; testA =0 ;
k = 100; %training image per iteration
n1 = 10;    %first layer neurons
%n2 = 10;    %second layer neurons
N1 = zeros(f0,n1);   %initialize weights
B1 = zeros(1,n1);
p = 0.2; %regularization

%for plots
XE_p = []; trainA_p = []; testA_p = []; XEtest_p = [];
xiter = []; xepoch = [];frame = []; N1_p =[]; B1_p = [];
figure('Position',[0 200 2000 400])

%end plots
tic
while iter<200 && testA<0.9
    epoch = iter/ceil(n0/k);
    xiter = [xiter iter];
    %segmentation
    
    %k1 = rem(n0,k); %final iteration if any, ideally 0
    k2= rem(iter,ceil(n0/k));
    imgset = trainimg(1+k2*k:min(n0,(k2+1)*k),:);
    labelset = trainlabel(1+k2*k:min(n0,(k2+1)*k));
    B1b = repmat(B1,k,1);
    X1 = imgset*N1+B1b;  %n0*n1
    Y1 = exp(X1)./repmat(sum(exp(X1),2),1,n1);  %softmax
    XE = -sum(log(Y1(sub2ind(size(Y1),(1:k)',labelset+1))))/k +(0.5*p/k)*sum(sum(N1.^2)); %cross entropy
    [~,idx] = max(Y1,[],2);
    trainA = numel(find(idx==labelset+1))/k;    %training accuracy
    trainA_p = [trainA_p trainA];
    XE_p = [XE_p XE];
    
    dXEdN1=zeros(f0,n1); dXEdB1 = zeros(1,n1);
    img_ph = zeros(k,n1);
    img_ph(sub2ind(size(img_ph),(1:k)',labelset+1)) = 1;
    
    %for j = 1:784   
    %    dXEdW_ph = repmat(imgset(:,j),1,n1).*(Y1 - img_ph);
    %    dXEdN1(j,:) = sum(dXEdW_ph);
    %end
    
    %vectorized
    dXEdW_ph = permute(repmat(imgset,1,1,10),[1 3 2]).*repmat(Y1-img_ph,1,1,784);
    dXEdN1 = permute(sum(dXEdW_ph),[3 2 1]);    
    
    dXEdB1 = sum(Y1-img_ph);
    
    dXEdN1 = dXEdN1 + (p/k).*N1;  % regularization
    
   % if rem(iter,10)==0
        % try on test data
        B1t = repmat(B1,ntest,1);
        Xtest = testimg*N1+B1t;
        Ytest = exp(Xtest)./repmat(sum(exp(Xtest),2),1,n1);
        XEtest = -sum(log(Ytest(sub2ind(size(Ytest),(1:ntest)',testlabel+1))))/ntest + (0.5*p/ntest)*sum(sum(N1.^2));
        [~,idxtest]=max(Ytest,[],2);
        testA = numel(find(idxtest==testlabel+1))/ntest;
        testA_p = [testA_p testA];
        XEtest_p = [XEtest_p XEtest];
        %xepoch = [xepoch epoch*ceil(n0/k)];
        xepoch = [xepoch iter];
        
        testimgplot = zeros(28*30,28*30);
        for i= 1:30
            testimgplot_ph=[];
            for j = 1:30
                testimgplot_ph= [testimgplot_ph reshape((testimg((i-1)*30+j,:).^(1/100)).*255,28,28)];
            end
            testimgplot(1+28*(i-1):28*i,:)=testimgplot_ph;
        end
        
  %  end
    
    gr = 0.002*exp(-iter)+0.001;
    %gr = 0.003;
    B1 = B1 - gr*dXEdB1;
    N1 = N1 - gr*dXEdN1;
    
    subplot(3,2,2)
    plot(xiter,XE_p)
    ylim([0 4])
    hold on
    plot(xepoch,XEtest_p)
    title('Cross Entropy')
    hold off
    
    subplot(3,2,1)
    plot(xiter,trainA_p)
    ylim([0 1])
    hold on
    plot(xepoch,testA_p)
    fplot(0.9)
    title('Accuracy');
    hold off
    
    subplot(3,2,3)
    N1_ph = reshape(N1,1,numel(N1));
    N1_p = [N1_p [max(N1_ph) ; min(N1_ph) ; mean(N1_ph); mean(N1_ph)+std(N1_ph) ; mean(N1_ph)-std(N1_ph)]];
    plot(xiter,N1_p)
    title('Weights');
    
    subplot(3,2,4)
    B1_p = [B1_p [max(B1) ; min(B1) ; mean(B1) ; mean(B1)+std(B1) ; mean(B1)- std(B1)]];
    plot(xiter,B1_p)
    title('Biases');
    
    wplot = zeros(2*28,5*28);
        N1b = N1;
        N1b = max(N1b,0);
        N1b = (N1b - repmat(mean(N1b),784,1))./repmat(std(N1b),784,1);
        N1b = max(N1b,0);
        %N1b = 1./(1+exp(-N1b*100));
            
    for i=1:2;
        wplot_ph =[];
        for j = 1:5;
            wplot_ph = [wplot_ph reshape((N1b(:,((i-1)*5+j)))'.*255,28,28)];
        end
        wplot(1+28*(i-1):28*i,:)=wplot_ph;
    end
    %wplot=max(wplot,0)
    wplot_ph = zeros(2*28,5*28,3);
    %wplot_ph(:,:,1) = wplot;
    %wplot_ph(:,:,2) = wplot;
    wplot_ph(:,:,3) = wplot;
    subplot(3,2,5:6)
    image(wplot_ph);
    
    imgplot = zeros(280,280);

    for i= 1:10
        imgplot_ph=[];
        for j = 1:10
            imgplot_ph= [imgplot_ph reshape((1-imgset((i-1)*10+j,:).^(1/1000)).*255,28,28)];
        end
        imgplot(1+28*(i-1):28*i,:)=imgplot_ph;
    end

    %imgplot_ph = zeros(280,280,3);
    %imgplot_ph(:,:,1) = imgplot;
    %imgplot_ph(:,:,2) = imgplot;
    %imgplot_ph(:,:,3) = imgplot;
    %imgplot_ph(1:56,:,2:3) = 0; recolour wrong prediction
  
    %subplot(2,3,3)
    %image(imgplot_ph);
    %title('Training Digits');
    
    %testimgplot_ph = zeros(28*30,28*30,3);
    %testimgplot_ph(:,:,1)= testimgplot;
    %testimgplot_ph(:,:,2)= testimgplot;
    %testimgplot_ph(:,:,3)= testimgplot;
    
    %subplot(2,3,6)
    %image(testimgplot_ph);
    %title('Test Digits');
    
    frame = [frame getframe];
    
    iter = iter + 1;
    
end
toc