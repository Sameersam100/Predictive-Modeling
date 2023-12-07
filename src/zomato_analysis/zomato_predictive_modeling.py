import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


class DataProcessor:
    """Processes data for analysis."""

    def __init__(self, filepath, encoding="ISO-8859-1"):
        """
        Initializes the DataProcessor with the dataset's path and encoding.

        Parameters:
        - filepath: str, path to the dataset.
        - encoding: str, file encoding type.
        """
        self.filepath = filepath
        self.encoding = encoding
        self.data = None
        self.preprocessor = None
        self.load_data()

    def load_data(self):
        """Loads data from the given file path."""
        self.data = pd.read_csv(self.filepath, encoding=self.encoding)
        self.clean_data()

    def clean_data(self):
        """Cleans and preprocesses the data."""
        self.data = self.data.drop(
            columns=[
                "Restaurant ID",
                "Restaurant Name",
                "Address",
                "Locality",
                "Locality Verbose",
                "Switch to order menu",
                "Rating color",
                "Rating text",
            ]
        )

    def preprocess_data(self):
        """Preprocesses the data for modeling."""
        numeric_features = (
            self.data.select_dtypes(include=["int64", "float64"])
            .drop("Aggregate rating", axis=1)
            .columns
        )
        categorical_features = self.data.select_dtypes(include=["object"]).columns

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return self.preprocessor


class ModelBuilder:
    """Builds and evaluates a predictive model."""

    def __init__(self, data, preprocessor):
        """
        Initializes the ModelBuilder with data and preprocessing pipeline.

        Parameters:
        - data: DataFrame, the dataset to model.
        - preprocessor: Pipeline, preprocessing steps to prepare the data.
        """
        self.data = data
        self.preprocessor = preprocessor
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def split_data(self, test_size=0.2, random_state=0):
        """Splits the data into training and testing sets."""
        X = self.data.drop("Aggregate rating", axis=1)
        y = self.data["Aggregate rating"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def build_model(self):
        """Builds a linear regression model pipeline."""
        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", LinearRegression()),
            ]
        )

    def train_model(self):
        """Trains the predictive model."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluates the model and prints out the metrics."""
        self.y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)
        r2 = r2_score(self.y_test, self.y_pred)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R-squared:", r2)

    def build_random_forest_model(self, n_estimators=100, random_state=0):
        """Builds a random forest regression model pipeline."""
        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=n_estimators, random_state=random_state
                    ),
                ),
            ]
        )

    def cross_validate_model(self, cv=5, scoring="neg_mean_squared_error"):
        """Performs cross-validation on the model and prints the results."""
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=cv, scoring=scoring
        )
        print("Cross-Validated MSE Scores:", -cv_scores)
        print("Average MSE:", -cv_scores.mean())
