from . import config
from .config import (logger)
from dotenv import load_dotenv
import os

# Define ANSI color codes
GREEN = "\033[32m"
WHITE = "\033[37m"
BRIGHT_BLUE = "\033[94m"
RESET = "\033[0m"  # Resets color to default

def reload_trade_variables():
    """
    Reload environment variables from .env and update trade-related variables.
    Returns a list of (name, formatted_value) tuples.
    """

    # Create updated trade_variables list with the same structure and formatting
    return [
        # Buy Criteria
        ("Portfolio Value", f"{config.config.PORTFOLIO_VALUE:,.2f}€"),
        ("Price Increase Threshold", f"{config.config.PRICE_INCREASE_THRESHOLD:.3f}%"),
        ("Min Volume EUR", f"{config.config.MIN_VOLUME_EUR:,.2f}€"),
        ("Allocation Per Trade", f"{config.config.ALLOCATION_PER_TRADE:.3f}"),
        ("Min Total Score", f"{config.config.MIN_TOTAL_SCORE:.2f}"),
        ("Max Slippage Buy", f"{config.config.MAX_SLIPPAGE_BUY:.3f}%"),
        ("Max Slippage Sell", f"{config.config.MAX_SLIPPAGE_SELL:.3f}%"),
        # Processing and Selling Criteria
        ("Concurrent Requests", f"{config.config.CONCURRENT_REQUESTS}"),
        ("Asset Threshold", f"{config.config.ASSET_THRESHOLD}"),
        ("Max Active Assets", f"{config.config.MAX_ACTIVE_ASSETS}"),
        ("Trailing Stop Factor", f"{config.config.TRAILING_STOP_FACTOR:.2f}"),
        ("Trailing Stop Factor Early", f"{config.config.TRAILING_STOP_FACTOR_EARLY:.2f}"),
        ("Adjusted Profit Target", f"{config.config.ADJUSTED_PROFIT_TARGET:.4f}"),
        ("Profit Target", f"{config.config.PROFIT_TARGET:.4f}"),
        # Tertiary Trade Variables
        ("Buy Fee", f"{config.config.BUY_FEE:.4f}"),
        ("Sell Fee", f"{config.config.SELL_FEE:.4f}"),
        ("Cat Loss Threshold", f"{config.config.CAT_LOSS_THRESHOLD:.3f}%"),
        ("Momentum Confirm Minutes", f"{config.config.MOMENTUM_CONFIRM_MINUTES}"),
        ("Momentum Threshold", f"{config.config.MOMENTUM_THRESHOLD:.3f}%"),
        ("Time Stop Minutes", f"{config.config.TIME_STOP_MINUTES}"),
        ("Min Holding Minutes", f"{config.config.MIN_HOLDING_MINUTES}"),
    ]

def print_trade_variables(vars_per_line=3, total_line_width=120):
    """
    Print trade-related variables with a specified number of variables per line.
    Variable names are in green, values in white, centered under the names.
    Each column is centered within its allocated portion of total_line_width.
    Reloads .env file to reflect changes during runtime.
    
    Args:
        vars_per_line (int): Number of variables to display per line.
        total_line_width (int): Total width of each line for centering.
    """
    # Reload trade variables to get the latest .env values
    trade_variables = reload_trade_variables()

    # Print headline, centered within total_line_width
    headline = "TRADE STRATEGY VARIABLES"
    logger.info(f"{BRIGHT_BLUE}{headline.center(total_line_width)}{RESET}")
    logger.info('')

    # Determine the maximum width needed for variable names and values
    max_name_width = max(len(name) for name, _ in trade_variables) + 4  # Add padding
    max_value_width = max(len(value) for _, value in trade_variables) + 4  # Add padding

    # Calculate column width as the maximum of name and value width
    col_width = max(max_name_width, max_value_width)

    # Calculate the width allocated to each column based on vars_per_line
    allocated_col_width = total_line_width // max(1, vars_per_line)

    # Ensure allocated_col_width is at least col_width to avoid truncation
    if allocated_col_width < col_width:
        logger.warning(
            f"Allocated column width ({allocated_col_width}) is less than required ({col_width}). "
            f"Increasing total_line_width to {col_width * vars_per_line}."
        )
        total_line_width = col_width * vars_per_line
        allocated_col_width = total_line_width // max(1, vars_per_line)

    # Process variables in chunks based on vars_per_line
    for i in range(0, len(trade_variables), vars_per_line):
        # Get the current chunk of variables
        chunk = trade_variables[i:i + vars_per_line]

        # Build variable names line (green)
        name_line = ""
        for name, _ in chunk:
            name_line += f"{GREEN}{name.center(allocated_col_width)}{RESET}"
        
        # Build values line (white)
        value_line = ""
        for _, value in chunk:
            value_line += f"{WHITE}{value.center(allocated_col_width)}{RESET}"

        # Log the lines
        logger.info(name_line)
        logger.info(value_line)
        logger.info('')

