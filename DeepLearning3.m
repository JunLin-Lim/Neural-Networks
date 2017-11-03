function weights = DeepLearning3(img,label)
n = numel(label)
X = img;

%intial weights and biases
W = rand(784,10);
S = repmat(sum(W),784,1);
W = W./S;

b = repmat(rand(1,10),n,1);

L = X*W + b

Y = softmax(L')'
Yp = zeros(n,10)
k = sub2ind([n 10],(1:n)',label+1)
Yp(k) = 1

E = -1.*sum(sum(Y.*log(Y)))

