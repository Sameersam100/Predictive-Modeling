from setuptools import setup, find_packages

setup(
    name="src.zomato_analysis",
    version="0.1",
    packages=["zomato_analysis"],
    package_dir={"zomato_analysis": "src/zomato_analysis"},
    description="A simple package for Zomato restaurants data analysis",
    author="Abdul Sameer Mohammed",
    author_email="amohamm8@mail.yu.edu",
    keywords=["zomato", "data analysis", "predictive modeling"],
    url="https://github.com/Sameersam100/Predictive-Modeling/",
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
