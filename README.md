# Auto Price Prediction Project
## This Project is a sample procedure to use Python machine learning regression model to predict car price from a sample data.

**Module 1:** [Exploratory Data Analysis](./exploratoryDataAnalysis.py)
- Used summary statistics to understand the basics of a data set.
- Used several types of plots to display distributions.
- Created scatter plots with different transparency.
- Used density plots and hex bin plots to overcome overplotting.
- Applied aesthetics to project additional dimensions of categorical and numeric variables onto a 2d plot surface.
- Used pair-wise scatter plots and conditioned plots to create displays with multiple axes.

**Module 2:** [Data Preparation](./dataPreparation.py)

Typical process:
- Visualization  of the dataset to understand the relationships and identify possible problems with the data.
- Data cleaning and tranformation to address the problems identified. It many cases, step 1 is then repeated to verify that the cleaning and transformation had the desired effect.
- Construction and evaluation of a machine learning models. Visualization of the results will often lead to understanding of further data preparation that is required; going back to step 1.

**Module 3:** [Apply Regression Model](./applyingLinearRession.py)
- Transformed the label value to make it more symmetric and closer to a Normal distribution.
- Aggregated categories of a categorical variable to improve the statistical representation.
- Scaled the numeric features.
- Recoded the categorical features as binary dummy variables.
- Fit the linear regression model using scikit-learn.
- Evaluated the performance of the model using both numeric and graphical methods.
                    
