function weights = Deep_Learning_01(img,label)
A = ones(20,785);
%A = nx785 matrix,

At = A(1:end,1:end-1);
Bt = A(1:end,end);
N = zeros(784,10);


%while WSmax < 0.9    

WS =img*N;

WSmax = max(transpose(WS))';