# MoLE-EN
Source code of the Regularized Mixture of Linear Experts

Paper: A Regularized Mixture of Linear Experts for Quality Prediction in Multimode and Multiphase Industrial Processes
       https://doi.org/10.3390/app11052040

# Description
The MoLE-EN can be used to derive the MoLE-Lasso and MoLE-RR as well. First, <img src="https://render.githubusercontent.com/render/math?math=\alpha = 1"> is considered to derive the MoLE-Lasso, then <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0.5"> is used to derive the MoLE-EN, and <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0"> is used to derive the MoLE-RR.

# Source Code
The source code presents the main class for MoLE-EN. The source code is provided in Matlab format. The usage is quite simple. Below an example.

First, you need to define the following paramaters first
```matlab

lambda=1e-3; % regularization parameter, 1e-3 is the default value;
ne=2; % number of experts, the default value is 2
iterations=50; % number of iterations of EM algorithm.
reg = 'lasso'; % type of regularizaion, the types are 'lasso' (for lasso penalty), 'en' (for elastic-net penalty), 'rr' (for ridge regression penalty).

% generate artificial data
pd = makedist('Normal');
samples = 1000;
x = random(pd,samples,1);
train = ceil(samples/2);
test = samples-train;
x_irre = random(pd,samples,2); % add two irrelevant variables

y(x<0.5,:) = x(x<0.5) .* 2 + ( randn(size(x(x<0.5))) .* 0.1 );
y(x>=0.5,:) = (2 - (x(x>=0.5) .* 2)) + ( randn(size(x(x>=0.5))) * 0.1 );
X_train = [x(1:train,:)  x_irre(1:train,:) ];
X_test  = [x(test+1:end,:) x_irre(1:train,:)];
Y_train = y(1:train,:);
Y_test  = y(test+1:end,:);

% initialize  model
mole_en = MoLE(ne,iterations,reg,lambda);
mole_en = mole_en.fit(X_train,Y_train);

Y_est = mole_en.predict(X_test);

% Plot results
plot(X_test(:,1),Y_test,'o')
hold on
plot(X_test(:,1),Y_est,'ro')
legend('real','estimated');
xlabel('samples')
ylabel('y-values')

```

It should output something like this:

![MoLE-EN results](/mole_en_output.png)

# Utilization

# Contact
f.souza@science.ru.nl
