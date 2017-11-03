function K = deeplearning2(a0,a1)
a0=a0';
%a0 = a0(1:500,:);      %first 500
%a1 = a1(1:500);        %first 500
%a0 = zeros(100,784); %img
%a1 = zeros(100,1);   %label
n = numel(a1); %no. images
gr = 0.03; %growth rate
U = 100;
a2 = zeros(n,10);
a2(sub2ind([n,10],(1:n)',a1+1)) = 1;

a5 =a0; a6 =[];
for i = 1:10
a5(a1 ~= i-1,:)= [];
a6 = [a6; sum(a5)];
a5 = a0;
end
a6 = a6';

M = rand(784,10);    %intitial weights 
b = rand(1,10);
b1 = repmat(b,n,1); %intital bias
w0 =a0*M+b1;            %weighted sum
w0 = w0./repmat(sum(w0,2),1,10);

E = -sum(sum(a2.*log(w0)));

    [~,a3] = max(w0,[],2);
    A = numel(find(a3-1==a1))/n;
    
iter = 0;
frame = [];E1=[E];x=[0];A1=[A];
%set(figure,'position',[0 0 2000 2000])
%fig = figure('position',[100 100 850 600]);
while  iter<10000
    %gr=0.0029*exp(-0.005*iter)+0.0001;
    dEdb = zeros(1,10);
    dEdW = zeros(784,10);
    %P = rem(iter,n/U);
    %a8 = a0(1+P*U:U*(P+1),:);%train U img
    %a9 = a1(1+P*U:U*(P+1));
    %a7 = a2(1+P*U:U*(P+1),:);
    
    
    a8 = a0; a9=a1; a7=a2; U = n;%train all
    
    for i = 1:10
        a5 = a8;            %img
    a5(a9 ~= i-1,:)= [];    %no. i img
    m = size(a5,1);
    w5 = a5*M+repmat(b,m,1);   %calc weights
    w6 = w5./repmat(sum(w5,2),1,10);    %normalize
    
    dEdb(i)= sum(w5(:,i).^(-1)) - sum(sum(w5,2).^(-1));
    
    dEdW(:,i)= sum(a5./repmat(w5(:,i),1,784))' - sum(a5./repmat(sum(w5,2),1,784))';
    
    end

    M = M + gr*dEdW;
    b = b + gr*dEdb;
    w2 = a8*M+repmat(b,U,1);
    w2 = w2./repmat(sum(w2,2),1,10);
    
    iter = iter + 1;
    E = -sum(sum(a7.*log(w2)));
    [~,a3] = max(w2,[],2);
    A = numel(find(a3-1==a9))/U;
    
    x = [x iter];
    E1 = [E1 E];
    A1 = [A1 A];
    
    subplot(2,2,1)
    plot(x,E1)
    title('Cross Entropy')
    subplot(2,2,2)
	plot(x,A1)
    title('Training Accuracy'); 
    ylim([0 1])
    subplot(2,2,3)
    plot(x,E1)
    title('Cross Entropy Period')
    xlim([max(0,iter-25) iter])
    subplot(2,2,4)
    plot(x,A1)
    title('Training Accuracy Period')
    ylim([A-0.05*A A+0.05*A])
    xlim([max(0,iter-25) iter])
       
    frame = [frame getframe];
end
end


    

