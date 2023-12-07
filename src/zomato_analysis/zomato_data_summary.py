import pandas as pd


class ZomatoDatasetLoader:
    """
    A class to load and retrieve the Zomato dataset.

    This class provides methods to load the Zomato dataset from a CSV file into a pandas DataFrame and retrieve it for analysis.

    Attributes
    ----------
    file_path : str
        The file path to the Zomato dataset CSV file.
    zomato_dataset : pandas.DataFrame or None
        The loaded Zomato dataset, or None if the dataset hasn't been loaded yet.

    Methods
    -------
    load_dataset():
        Loads the Zomato dataset from the CSV file and stores it in zomato_dataset attribute.
    get_dataset():
        Returns the loaded Zomato dataset if available, otherwise returns an error message.
    """

    def __init__(self, file_path):
        """
        Constructs all the necessary attributes for the ZomatoDatasetLoader object.

        Parameters
        ----------
            file_path : str
                The file path to the Zomato dataset CSV file.
        """
        self.file_path = file_path
        self.zomato_dataset = None

    def load_dataset(self):
        """
        Load the Zomato dataset.

        Attempts to read the Zomato dataset CSV file and store it in the zomato_dataset attribute.
        Handles FileNotFoundError if the file is not found at the given path.

        Returns
        -------
        pandas.DataFrame or str
            The loaded Zomato dataset DataFrame, or an error message if the file is not found.
        """
        try:
            self.zomato_dataset = pd.read_csv(self.file_path, encoding="ISO-8859-1")
            return self.zomato_dataset
        except FileNotFoundError:
            return f"Error: File not found at {self.file_path}"

    def get_dataset(self):
        """
        Retrieve the loaded Zomato dataset.

        Returns the Zomato dataset if it has been loaded. If the dataset is not loaded, returns an error message.

        Returns
        -------
        pandas.DataFrame or str
            The loaded Zomato dataset DataFrame, or an error message if the dataset is not loaded.
        """
        if self.zomato_dataset is not None:
            return self.zomato_dataset
        else:
            return "Error: Zomato dataset not loaded. Use load_dataset() method first."


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataSummary:
    """
    A class for providing a comprehensive summary of a dataset.

    This class includes various methods to display information about a dataset, aiding in understanding its structure, content, and potential issues before data cleaning, preprocessing, and exploratory data analysis.

    Attributes:
    ----------
    dataset : pandas.DataFrame
        The dataset to be summarized.

    Methods:
    -------
    display_head(n=5):
        Displays the first n rows of the dataset.

    display_shape():
        Shows the number of rows and columns in the dataset.

    display_info():
        Provides a concise summary of the dataset.

    display_descriptive_statistics():
        Shows descriptive statistics for numerical columns.

    display_missing_values():
        Indicates the number of missing values in each column.

    display_unique_values(column):
        Shows unique values in a specified column.

    display_data_types():
        Displays the data types of each column.

    plot_missing_values():
        Visualizes the distribution of missing values in the dataset.
    """

    def __init__(self, dataset):
        """
        Initializes the DataSummary with the specified dataset.
        """
        self.dataset = dataset

    def display_head(self, n=5):
        """
        Displays the first n rows of the dataset.
        """
        return self.dataset.head(n)

    def display_shape(self):
        """
        Displays the number of rows and columns in the dataset.
        """
        return self.dataset.shape

    def display_info(self):
        """
        Provides a concise summary of the dataset.
        """
        return self.dataset.info()

    def display_descriptive_statistics(self):
        """
        Shows descriptive statistics for numerical columns.
        """
        return self.dataset.describe()

    def display_missing_values(self):
        """
        Indicates the number of missing values in each column.
        """
        return self.dataset.isna().sum()

    def display_unique_values(self, column):
        """
        Shows unique values in a specified column.
        """
        return self.dataset[column].unique()

    def display_data_types(self):
        """
        Displays the data types of each column.
        """
        return self.dataset.dtypes

    def plot_missing_values(self):
        """
        Visualizes the distribution of missing values in the dataset.
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            self.dataset.isnull(), cbar=False, yticklabels=False, cmap="viridis"
        )
        plt.title("Distribution of Missing Values")
        plt.show()
