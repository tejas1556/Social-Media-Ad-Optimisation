
Maximizing ROI for Social Media Ad Campaigns

Overview
This project focuses on using optimization techniques to maximize the Return on Investment (ROI) for a social media advertising campaign. Several optimization techniques are employed to determine the optimal values for key features like Engagement Score, Clicks, Impressions, and Channel Used.

The goal is to maximize ROI based on these features while respecting constraints derived from the dataset.

Techniques Used

1. Simplex Method
   - Objective: Maximize the ROI using linear programming.
   - Variables: Engagement Score, Clicks, Impressions, Channel Used.
   - Constraints: Based on dataset averages and logical relations (e.g., Clicks ≤ Impressions).

2. Genetic Algorithm
   - Objective: Maximize the ROI using an evolutionary optimization technique.
   - Process: Used population-based search methods to find the optimal combination of decision variables.
   - Goal: Determine the best combination of Engagement Score, Clicks, Impressions, and Channel Used to maximize ROI.

3. Bayesian Optimization
   - Objective: Optimize hyperparameters of a machine learning model to predict and maximize ROI.
   - Process: Employed Bayesian Search to tune hyperparameters and find the optimal model configuration.

4. Linear Regression
   - Objective: Estimate the impact of Engagement Score, Clicks, Impressions, and Channel Used on ROI.
   - Process: Built a linear model to predict ROI and used its coefficients for further optimization.
   - Key Insight: A regression model helped derive coefficients to understand feature importance and optimize the campaign.

Files and Structure

- FINAL_AO_project.ipynb: The main Jupyter Notebook containing the code for optimization techniques.
- Social_Media_Advertising.csv: The dataset used for the analysis and optimization, which includes variables like Engagement Score, Clicks, Impressions, and ROI.
- README.md: This file.

Getting Started

Prerequisites
To run the project locally, make sure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- scipy
- deap (for Genetic Algorithm)
- skopt (for Bayesian Optimization)

You can install the necessary packages using:

pip install pandas numpy matplotlib scikit-learn xgboost scipy deap scikit-optimize

Running the Project
1. Load the dataset: Begin by loading Social_Media_Advertising.csv into a pandas dataframe.
2. Data Preprocessing: Perform necessary data cleaning (e.g., encoding categorical variables and handling missing values).
3. Optimization Techniques:
   - Run the Simplex Method to find optimal values for the decision variables.
   - Apply Genetic Algorithm to further maximize ROI by exploring a population of solutions.
   - Use Bayesian Optimization to fine-tune hyperparameters and improve predictive models.
4. Model Evaluation: Evaluate the models using metrics like MSE and R² to compare performance.
5. ROI Calculation: Based on the optimal feature values, calculate the maximized ROI.

Example Usage

# Load necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog

# Load the dataset
data = pd.read_csv('Social_Media_Advertising.csv')

# Preprocess data (Label encoding for categorical variables)
data['Channel_Used'] = data['Channel_Used'].apply(lambda x: 0 if x == 'Facebook' else 1)

# Extract features and target variable
X = data[['Engagement_Score', 'Clicks', 'Impressions', 'Channel_Used']]
y = data['ROI']

# Apply Simplex Method for Optimization
c = [-5, -1.5, -0.01, 10]  # Coefficients for the objective function (negative for maximization)
A = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0]]  # Constraints matrix
b = [6, 100, 5000, 1, 0]  # Upper bounds for the constraints
result = linprog(c, A_ub=A, b_ub=b)

print(f"Optimal Solution: {result.x}")
print(f"Maximized ROI: {result.fun * -1}")  # Revert back to positive ROI

Results and Discussion

- The Simplex Method yields an optimal solution, determining the best combination of Engagement Score, Clicks, Impressions, and Channel Used to maximize ROI.
- Genetic Algorithm explores multiple solutions through an evolutionary process and finds near-optimal solutions for maximizing ROI.
- Bayesian Optimization fine-tunes hyperparameters for the predictive models, leading to better model performance and maximizing ROI.

Future Improvements
- Advanced Optimization Algorithms: Techniques like Simulated Annealing, Ant Colony Optimization, or Particle Swarm Optimization can be explored for further fine-tuning of ROI maximization.
- Real-Time Data Integration: Integrating real-time data could allow for dynamic campaign adjustments.
- Business Constraints: In future implementations, more detailed business-specific constraints can be incorporated to reflect real-world ad budget limits and performance factors.

License

This project is licensed under the MIT License - see the LICENSE.md file for details.
