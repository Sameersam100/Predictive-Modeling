import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import plotly.express as px


class DataCleaning:
    """
    A class for performing basic data cleaning tasks on a dataset without using sklearn features.

    This class provides methods for handling missing values, removing duplicates, manually encoding categorical variables, and normalizing numerical features.

    Attributes:
    ----------
    dataset : pandas.DataFrame
        The dataset to be cleaned.

    Methods:
    -------
    handle_missing_values(strategy='mean'):
        Fills or drops missing values based on the specified strategy.

    remove_duplicates():
        Removes duplicate rows from the dataset.

    encode_categorical_variables(columns):
        Manually encodes the specified categorical columns.

    normalize_features(columns):
        Normalizes numerical features.
    """

    def __init__(self, dataset):
        """
        Initializes the DataCleaning with the specified dataset.
        """
        self.dataset = dataset

    def handle_missing_values(self, strategy="mean"):
        """
        Fills or drops missing values based on the specified strategy.
        """
        if strategy == "drop":
            self.dataset.dropna(inplace=True)
        else:
            for col in self.dataset.select_dtypes(include=["float64", "int64"]).columns:
                if strategy == "mean":
                    fill_value = self.dataset[col].mean()
                elif strategy == "median":
                    fill_value = self.dataset[col].median()
                elif strategy == "mode":
                    fill_value = self.dataset[col].mode()[0]
                self.dataset[col].fillna(fill_value, inplace=True)

    def remove_duplicates(self):
        """
        Removes duplicate rows from the dataset.
        """
        self.dataset.drop_duplicates(inplace=True)

    def encode_categorical_variables(self, columns):
        """
        Manually encodes the specified categorical columns.
        """
        for col in columns:
            dummies = pd.get_dummies(self.dataset[col], prefix=col)
            self.dataset = pd.concat([self.dataset, dummies], axis=1)
            self.dataset.drop(col, axis=1, inplace=True)

    def normalize_features(self, columns):
        """
        Normalizes numerical features.
        """
        for col in columns:
            self.dataset[col] = (
                self.dataset[col] - self.dataset[col].mean()
            ) / self.dataset[col].std()


class EnhancedEDA:
    """Performs enhanced exploratory data analysis on the dataset."""

    def __init__(self, data):
        """
        Initializes the EnhancedEDA with the dataset.

        Parameters:
        - data: DataFrame, the dataset to analyze.
        """
        self.data = data

    def display_head(self):
        """Displays the first few rows of the dataframe."""
        return self.data.head()

    def show_info(self):
        """Prints information about the dataframe including data types and missing values."""
        return self.data.info()

    def plot_histogram(self, column):
        """
        Plots a histogram for a specified column with mean line.

        Parameters:
        - column: str, the column name for which the histogram is to be plotted.
        """
        # Set the style of seaborn
        sns.set(style="whitegrid")

        # Plotting the histogram using seaborn
        plt.figure(figsize=(10, 6))  # Increase the size of the plot
        sns.histplot(self.data[column], kde=True, color="skyblue")

        # Plot a vertical line for the mean
        plt.axvline(
            self.data[column].mean(), color="red", linestyle="dashed", linewidth=2
        )

        # Add a title and labels
        plt.title("Distribution of " + column, fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)

        # Add text to show the mean value
        plt.text(
            x=self.data[column].mean(),
            y=max(plt.ylim()),
            s="Mean",
            color="red",
            fontsize=12,
        )

        # Show the plot
        plt.show()

    def plot_rating_distribution(
        self,
        column="Aggregate rating",
        color="skyblue",
        title_fontsize=16,
        label_fontsize=14,
    ):
        """
        Plots a histogram for the specified column with a KDE and mean line.

        Parameters:
        - column: str, the column name for which the histogram is to be plotted.
        - color: str, color for the histogram bars.
        - title_fontsize: int, fontsize for the histogram title.
        - label_fontsize: int, fontsize for the x and y labels.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], kde=True, color=color)
        plt.axvline(
            self.data[column].mean(), color="red", linestyle="dashed", linewidth=2
        )
        plt.title(f"Distribution of {column}", fontsize=title_fontsize)
        plt.xlabel(column, fontsize=label_fontsize)
        plt.ylabel("Frequency", fontsize=label_fontsize)
        plt.text(
            x=self.data[column].mean(),
            y=max(plt.ylim()),
            s="Mean",
            color="red",
            fontsize=label_fontsize,
        )
        plt.show()

    def find_top_cuisines(self, num_top_cuisines=10):
        """
        Identifies and prints the most common cuisines.

        Parameters:
        - num_top_cuisines: int, the number of top cuisines to return.
        """
        top_cuisines = self.data["Cuisines"].value_counts().head(num_top_cuisines)
        print("Most Common Cuisines:\n", top_cuisines)
        return top_cuisines

    def average_rating_by_cuisine(self):
        """
        Calculates and prints the average aggregate rating by cuisine.
        """
        average_rating = (
            self.data.groupby("Cuisines")["Aggregate rating"]
            .mean()
            .sort_values(ascending=False)
        )
        print("Average Rating by Cuisine:\n", average_rating.head(10))
        return average_rating.head(10)

    def plot_online_delivery_rating(self):
        """Plots boxplot comparing aggregate ratings with online delivery availability."""
        sns.boxplot(x="Has Online delivery", y="Aggregate rating", data=self.data)
        plt.title("Aggregate Rating vs Online Delivery")
        plt.show()

    def plot_table_booking_rating(self):
        """Plots boxplot comparing aggregate ratings with table booking availability."""
        sns.boxplot(x="Has Table booking", y="Aggregate rating", data=self.data)
        plt.title("Aggregate Rating vs Table Booking")
        plt.show()

    def top_cities_by_rating(self, num_top_cities=10):
        """Prints and returns the top cities with the highest average ratings."""
        top_cities = (
            self.data.groupby("City")["Aggregate rating"]
            .mean()
            .sort_values(ascending=False)
        )
        print("Top Cities by Average Rating:\n", top_cities.head(num_top_cities))
        return top_cities.head(num_top_cities)

    def plot_rating_distribution_by_city(self, city, bins=20):
        """Plots the rating distribution for a specific city."""
        sns.histplot(
            self.data[self.data["City"] == city]["Aggregate rating"], bins=bins
        )
        plt.title(f"Rating Distribution in {city}")
        plt.show()

    def plot_average_rating_by_price(self):
        """Calculates and plots the average aggregate rating by price range."""
        average_rating_by_price = self.data.groupby("Price range")[
            "Aggregate rating"
        ].mean()
        sns.barplot(x=average_rating_by_price.index, y=average_rating_by_price.values)
        plt.title("Average Aggregate Rating by Price Range")
        plt.xlabel("Price Range")
        plt.ylabel("Average Aggregate Rating")
        plt.show()

    def plot_votes_vs_rating(self):
        """Plots a scatter plot of Votes vs Aggregate Rating."""
        sns.scatterplot(x="Votes", y="Aggregate rating", data=self.data)
        plt.title("Votes vs Aggregate Rating")
        plt.show()

    def show_votes_rating_correlation(self):
        """Prints the correlation between Votes and Aggregate Rating."""
        print(
            "Correlation between Votes and Aggregate Rating:\n",
            self.data[["Votes", "Aggregate rating"]].corr(),
        )

    def plot_price_range_rating(self):
        """Plots a boxplot of Price Range vs Aggregate Rating."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Price range", y="Aggregate rating", data=self.data)
        plt.title("Aggregate Rating by Price Range")
        plt.xlabel("Price Range")
        plt.ylabel("Aggregate Rating")
        plt.show()

    def plot_top_cuisines_count(self):
        """Plots a count plot of the top 10 cuisines."""
        plt.figure(figsize=(12, 8))
        sns.countplot(
            y="Cuisines",
            data=self.data,
            order=self.data["Cuisines"].value_counts().index[:10],
        )
        plt.title("Top 10 Cuisines")
        plt.xlabel("Count")
        plt.ylabel("Cuisines")
        plt.show()

    def show_correlation_heatmap(self):
        """Displays a heatmap of the correlation matrix."""
        plt.figure(figsize=(10, 8))
        # Set numeric_only to True to compute correlation only for numeric columns
        corr = self.data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_pairwise_features(self):
        """Displays pairwise plots of key features."""
        sns.pairplot(
            self.data[
                ["Aggregate rating", "Average Cost for two", "Votes", "Price range"]
            ]
        )
        plt.suptitle("Pairwise Plots of Key Features")
        plt.show()

    def plot_average_rating_online_delivery(self):
        """Plots a bar plot for average aggregate rating by online delivery option."""
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Has Online delivery", y="Aggregate rating", data=self.data)
        plt.title("Average Aggregate Rating by Online Delivery Option")
        plt.xlabel("Has Online Delivery")
        plt.ylabel("Average Aggregate Rating")
        plt.show()

    def map_high_rated_restaurants(self, rating_threshold=4.5):
        """
        Creates a Folium map with markers for high-rated restaurants.

        Parameters:
        - rating_threshold: float, the minimum rating to filter high-rated restaurants.
        """
        high_rated = self.data[self.data["Aggregate rating"] >= rating_threshold]

        # Check if high-rated restaurants are available
        if high_rated.empty:
            print("No high-rated restaurants found.")
            return

        # Creating a base map
        base_map = folium.Map(
            location=[high_rated["Latitude"].mean(), high_rated["Longitude"].mean()],
            zoom_start=12,
        )

        # Creating a Marker Cluster
        marker_cluster = MarkerCluster().add_to(base_map)

        # Adding markers to the map
        for idx, row in high_rated.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"{row['Restaurant Name']}, Cuisine: {row['Cuisines']}, Rating: {row['Aggregate rating']}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(marker_cluster)

        return base_map

    def plot_cuisines_vs_rating_cost(self):
        """
        Creates a scatter plot showing the relationship between the number of cuisines offered,
        average rating, and cost in different cities.
        """

        def count_cuisines(cuisines):
            if isinstance(cuisines, str):
                return len(cuisines.split(", "))
            return 0

        self.data["Number of Cuisines"] = self.data["Cuisines"].apply(count_cuisines)

        city_grouped = (
            self.data.groupby("City")
            .agg(
                {
                    "Aggregate rating": "mean",
                    "Average Cost for two": "mean",
                    "Number of Cuisines": "mean",
                }
            )
            .reset_index()
        )

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=city_grouped,
            x="Number of Cuisines",
            y="Aggregate rating",
            size="Average Cost for two",
            hue="City",
            alpha=0.7,
            sizes=(20, 200),
        )
        plt.title(
            "Relationship Between Number of Cuisines, Average Rating, and Cost in Different Cities"
        )
        plt.xlabel("Average Number of Cuisines per Restaurant")
        plt.ylabel("Average Aggregate Rating")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()

    def plot_interactive_cuisines_vs_rating_cost(self):
        """
        Creates an interactive Plotly scatter plot showing the relationship
        between the number of cuisines, average rating, and cost in different cities.
        """

        def count_cuisines(cuisines):
            if isinstance(cuisines, str):
                return len(cuisines.split(", "))
            return 0

        self.data["Number of Cuisines"] = self.data["Cuisines"].apply(count_cuisines)

        city_grouped = (
            self.data.groupby("City")
            .agg(
                {
                    "Aggregate rating": "mean",
                    "Average Cost for two": "mean",
                    "Number of Cuisines": "mean",
                }
            )
            .reset_index()
        )

        fig = px.scatter(
            city_grouped,
            x="Number of Cuisines",
            y="Aggregate rating",
            size="Average Cost for two",
            color="City",
            hover_name="City",
            size_max=60,
            title="Relationship Between Number of Cuisines, Average Rating, and Cost in Different Cities",
        )

        fig.update_layout(
            xaxis_title="Average Number of Cuisines per Restaurant",
            yaxis_title="Average Aggregate Rating",
            legend_title="City",
        )

        fig.show()
