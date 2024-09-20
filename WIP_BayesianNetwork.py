import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Example dataset
# Assume data is collected with 'TaskDifficulty' (0, 1, 2) and 'TaskSuccess' (1 for success, 0 for failure)
data = pd.DataFrame({
    'TaskDifficulty': [0, 1, 1, 2, 2, 2, 0, 1, 2, 0],
    'TaskSuccess': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
})

# Define the structure of the Bayesian network
# TaskDifficulty -> TaskSuccess
model = BayesianNetwork([('TaskDifficulty', 'TaskSuccess')])

# Estimate the CPDs (Conditional Probability Distributions) using Maximum Likelihood Estimation
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Perform inference on the network
inference = VariableElimination(model)

# Query the probability of task success given a specific difficulty (e.g., difficulty 2)
query_result = inference.query(variables=['TaskSuccess'], evidence={'TaskDifficulty': 2})
print(query_result)

