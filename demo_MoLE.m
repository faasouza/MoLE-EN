% DEMO to run the MoLE-EN with lasso penalty

% For use with MATLAB
% Copyright (C) 2021 -- Francisco Souza <alexandre.andry@gmail.com> or <f.souza@science.ru.nl>
% MoLE-EN comes with ABSOLUTELY NO WARRANTY;
% In case of application of this method in any publication,
% please, cite the original work:
% Souza, Francisco; Mendes, Jérôme; Araújo, Rui. 2021. 
% "A Regularized Mixture of Linear Experts for Quality Prediction in Multimode and Multiphase Industrial Processes" 
% Appl. Sci. 11, no. 5: 2040. 
% DOI: https://doi.org/10.3390/app11052040


clc;close all;clear

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