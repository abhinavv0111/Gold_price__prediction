# Gold_price__prediction## Overview

This project focuses on predicting gold prices using a RandomForestRegressor. The model is trained on historical gold price data and aims to provide insights into potential future prices.

## Project Structure

- [Notebook](Gold_Price_Prediction.ipynb): Jupyter Notebook containing the code for data exploration, preprocessing, model training, and evaluation.
- [Dataset](gld_price_data.csv): CSV file containing historical gold price data.

## Data Exploration

The initial analysis involves loading and exploring the dataset to understand its structure and relationships. Key steps include:

- Handling missing values.
- Exploring correlations between features.
- Visualizing the distribution of the target variable (GLD - Gold price).

## Model Training

The RandomForestRegressor from scikit-learn is used to train the prediction model. Key steps include:

- Data preprocessing: Extracting features and target variable, splitting into training and testing sets.
- Model training: Using the RandomForestRegressor to learn patterns from the historical data.

## Model Evaluation

The trained model is evaluated using the R-squared metric, providing insights into its predictive performance.

## Results

The project visualizes the actual vs. predicted gold prices, allowing for a qualitative assessment of the model's effectiveness.


## Usage

To run the code and reproduce the results, follow these steps:

1. Clone the repository:

    ```bash
https://github.com/abhinavv0111/Gold_price__prediction/edit/main/README.md
    ```

2. Open the Jupyter Notebook [Gold_Price_Prediction.ipynb] and run each cell sequentially.

## Dependencies

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
