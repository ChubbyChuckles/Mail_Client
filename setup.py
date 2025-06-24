from setuptools import find_packages, setup

setup(
    name="trading_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ccxt",
        "pandas",
        "python-dotenv",
        "pyarrow",
        "gspread",
        "oauth2client",
        "numpy",
        "python-telegram-bot",
        "tenacity",
        "pytest",
        "pytest-mock",
    ],
    author="Christian Rickert",
    description="A cryptocurrency trading bot using Bitvavo API",
    license="MIT",
)
