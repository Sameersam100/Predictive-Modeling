from setuptools import setup, find_packages

setup(
    name="src.zomato_analysis",
    version="0.1",
    packages=["zomato_analysis"],
    package_dir={"zomato_analysis": "src/zomato_analysis"},
    description="A simple package for Zomato restaurants data analysis",
    author="Sameer",
    author_email="your.email@example.com",
    keywords=["zomato", "data analysis", "predictive modeling"],
    url="https://github.com/yourusername/your_package_repo",
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "folium",
        "plotly",
    ],
)
