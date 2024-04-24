# from statsmodels.tsa.ardl import ARDL
# Econ\2. personal\DO\statsmodels-main\statsmodels\tsa\ardl\model.py

import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.ardl as ardl
np.random.seed(42)
X = np.random.rand(100)  # 100 random numbers between 0 and 1
beta_0 = 1.5  # intercept
beta_1 = 4.0  # slope
errors = np.random.normal(0, 1, 100)
Y = beta_0 + beta_1 * X + errors

X = sm.add_constant(X)  # adds a constant term to the predictor
model = ardl.ARDL(Y, X)
results = model.fit(method="quantreg")