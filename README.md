# Rental Price Prediction Model

This script is designed to predict rental prices based on a variety of features using a machine learning model. It preprocesses the data, trains a RandomForestRegressor model, and saves the trained model for later use.

## Dependencies

- pandas
- numpy
- scikit-learn
- joblib
- gc (Garbage Collector interface)

## Workflow

### Data Preprocessing
1. **Data Loading**: Loads data from `rent_new.csv` using specified columns.
2. **Data Cleaning**: Imputes missing values and encodes categorical variables.
3. **Feature Engineering**: Transforms and creates new features to improve model performance.
4. **Outlier Removal**: Drops outlier records based on certain conditions to ensure model robustness.

### Model Training
1. **Data Splitting**: Splits the data into training and testing sets.
2. **Model Initialization**: Initializes a RandomForestRegressor model with specific parameters.
3. **Model Training**: Trains the model on the training data.
4. **Performance Evaluation**: Evaluates the model using Median Absolute Percentage Error (MdAPE).

### Model Saving
- Saves the trained model to the disk using joblib for later use or deployment.

## Usage

1. Ensure all dependencies are installed.
2. Adjust the file paths and any model parameters as needed.
3. Run the script using a Python interpreter.

## Functions Description

- `mdape`: Calculates the Median Absolute Percentage Error.
- `imputer`: Imputes missing values in the dataset.
- `encoder`: Encodes categorical features using predefined dictionaries.
- `feature_transformator`: Transforms and creates new features.
- `drop_outliers`: Removes outliers from the dataset.
- `feature_extractor`: A comprehensive function that applies all preprocessing steps.
- `main`: The main function that executes the entire workflow.

## Notes

- Modify file paths and dictionaries according to your directory structure and data.
- Adjust model parameters as needed based on your specific dataset and performance requirements.
- The script assumes that certain preprocessed dictionaries for encoding are available and formatted as CSV files.

## Contribution

Feel free to fork this project and submit your contributions via pull requests.

