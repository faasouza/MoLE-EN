# MoLE-EN
Source code of the Regularized Mixture of Linear Experts;

Reference paper: Souza, Francisco; Mendes, Jérôme; Araújo, Rui. 2021. "A Regularized Mixture of Linear Experts for Quality Prediction in Multimode and Multiphase Industrial Processes" Appl. Sci. 11, no. 5: 2040. https://doi.org/10.3390/app11052040

# Description
The MoLE-EN can be used to derive the MoLE-Lasso and MoLE-RR as well. The usage is descripted below. You can also run the script below from demo_MoLE.m

# Source Code
The source code presents the main class for MoLE-EN. The source code is provided in Matlab format. The usage is quite simple. Below an example that runs on an artifical dataset with two irrelevant variables.


```matlab

lambda=1e-3; % regularization parameter, 1e-3 is the default value;
ne=2; % number of experts, the default value is 2
iterations=50; % number of iterations of EM algorithm.
reg = 'lasso'; % type of regularizaion, the types are 'lasso' (for lasso penalty), 'en' (for elastic-net penalty), 'rr' (for ridge regression penalty).

% generate artificial data
pd = makedist('Normal');
samples = 50;
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
% fit model
mole_en = mole_en.fit(X_train,Y_train);
% predict from test data
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
You can also run the script below from demo_MoLE.m


# Contact
If you have any question, bug report, please contact me trough the email: f.souza@science.ru.nl
