# trading_bot/src/trade_variables.py
import os
from dotenv import load_dotenv
from . import config
from .config import logger

# Define ANSI color codes
GREEN = "\033[32m"
WHITE = "\033[37m"
BRIGHT_BLUE = "\033[94m"
RESET = "\033[0m"  # Resets color to default

def reload_trade_variables():
    """
    Reload environment variables from .env and update trade-related variables.

    Returns:
        list: List of (name, formatted_value) tuples.

    Raises:
        OSError: If the .env file cannot be read.
    """
    try:
        # Explicitly reload .env file
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if not os.path.exists(env_path):
            logger.warning(f".env file not found at {env_path}. Using existing environment variables.")
        else:
            if not load_dotenv(env_path):
                logger.error(f"Failed to load .env file from {env_path}")
                send_alert("Environment Load Failure", f"Failed to load .env file from {env_path}")
                raise OSError(f"Failed to load .env file from {env_path}")

        # Required configuration variables with fallback values
        required_vars = {
            "PORTFOLIO_VALUE": (float, 10000.0, "non-negative float"),
            "PRICE_INCREASE_THRESHOLD": (float, 0.5, "non-negative float"),
            "MIN_VOLUME_EUR": (float, 1000.0, "non-negative float"),
            "ALLOCATION_PER_TRADE": (float, 0.1, "non-negative float"),
            "MIN_TOTAL_SCORE": (float, 0.7, "non-negative float"),
            "MAX_SLIPPAGE_BUY": (float, 0.015, "non-negative float"),
            "MAX_SLIPPAGE_SELL": (float, -0.015, "negative float"),
            "CONCURRENT_REQUESTS": (int, 5, "non-negative integer"),
            "ASSET_THRESHOLD": (int, 5, "non-negative integer"),
            "MAX_ACTIVE_ASSETS": (int, 10, "non-negative integer"),
            "TRAILING_STOP_FACTOR": (float, 0.05, "non-negative float"),
            "TRAILING_STOP_FACTOR_EARLY": (float, 0.03, "non-negative float"),
            "ADJUSTED_PROFIT_TARGET": (float, 0.015, "non-negative float"),
            "PROFIT_TARGET": (float, 0.02, "non-negative float"),
            "BUY_FEE": (float, 0.0025, "non-negative float"),
            "SELL_FEE": (float, 0.0025, "non-negative float"),
            "CAT_LOSS_THRESHOLD": (float, 0.08, "non-negative float"),
            "MOMENTUM_CONFIRM_MINUTES": (int, 5, "non-negative integer"),
            "MOMENTUM_THRESHOLD": (float, -0.2, "negative float"),
            "TIME_STOP_MINUTES": (int, 60, "non-negative integer"),
            "MIN_HOLDING_MINUTES": (int, 15, "non-negative integer"),
        }

        trade_variables = []
        invalid_vars = []

        # Validate and retrieve configuration variables
        for var_name, (expected_type, default_value, type_desc) in required_vars.items():
            try:
                value = getattr(config.config, var_name, None)
                if value is None:
                    logger.warning(f"Configuration variable {var_name} missing, using default: {default_value}")
                    value = default_value
                else:
                    value = expected_type(value)
                    if expected_type in (int, float) and type_desc == "non-negative float" and value < 0:
                        logger.warning(f"Invalid value for {var_name}: {value} (must be non-negative). Using default: {default_value}")
                        value = default_value
                    elif expected_type == int and type_desc == "non-negative integer" and (not isinstance(value, int) or value < 0):
                        logger.warning(f"Invalid value for {var_name}: {value} (must be non-negative integer). Using default: {default_value}")
                        value = default_value
                    elif expected_type == float and type_desc == "negative float" and value > 0:
                        logger.warning(f"Invalid value for {var_name}: {value} (must be negative). Using default: {default_value}")
                        value = default_value

                # Format the value for display
                if var_name in ["PORTFOLIO_VALUE", "MIN_VOLUME_EUR"]:
                    formatted_value = f"{value:,.2f}€"
                elif var_name in ["PRICE_INCREASE_THRESHOLD", "MAX_SLIPPAGE_BUY", "MAX_SLIPPAGE_SELL", "CAT_LOSS_THRESHOLD", "MOMENTUM_THRESHOLD"]:
                    formatted_value = f"{value:.3f}%"
                elif var_name in ["ALLOCATION_PER_TRADE"]:
                    formatted_value = f"{value:.3f}"
                elif var_name in ["MIN_TOTAL_SCORE", "TRAILING_STOP_FACTOR", "TRAILING_STOP_FACTOR_EARLY"]:
                    formatted_value = f"{value:.2f}"
                elif var_name in ["ADJUSTED_PROFIT_TARGET", "PROFIT_TARGET", "BUY_FEE", "SELL_FEE"]:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                trade_variables.append((var_name, formatted_value))
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid value for {var_name}: {getattr(config.config, var_name, 'N/A')} ({e}). Using default: {default_value}")
                invalid_vars.append(var_name)
                formatted_value = (
                    f"{default_value:,.2f}€" if var_name in ["PORTFOLIO_VALUE", "MIN_VOLUME_EUR"] else
                    f"{default_value:.3f}%" if var_name in ["PRICE_INCREASE_THRESHOLD", "MAX_SLIPPAGE_BUY", "MAX_SLIPPAGE_SELL", "CAT_LOSS_THRESHOLD", "MOMENTUM_THRESHOLD"] else
                    f"{default_value:.3f}" if var_name == "ALLOCATION_PER_TRADE" else
                    f"{default_value:.2f}" if var_name in ["MIN_TOTAL_SCORE", "TRAILING_STOP_FACTOR", "TRAILING_STOP_FACTOR_EARLY"] else
                    f"{default_value:.4f}" if var_name in ["ADJUSTED_PROFIT_TARGET", "PROFIT_TARGET", "BUY_FEE", "SELL_FEE"] else
                    str(default_value)
                )
                trade_variables.append((var_name, formatted_value))

        if invalid_vars:
            send_alert("Trade Variables Validation Warning", f"Invalid values for: {', '.join(invalid_vars)}. Used defaults.")

        logger.debug(f"Successfully reloaded {len(trade_variables)} trade variables")
        return trade_variables
    except OSError as e:
        logger.error(f"File operation error in reload_trade_variables: {e}", exc_info=True)
        send_alert("Trade Variables Reload Failure", f"File operation error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in reload_trade_variables: {e}", exc_info=True)
        send_alert("Trade Variables Reload Failure", f"Unexpected error: {e}")
        return []

def print_trade_variables(vars_per_line=3, total_line_width=120):
    """
    Print trade-related variables with a specified number of variables per line.
    Variable names are in green, values in white, centered under the names.
    Each column is centered within its allocated portion of total_line_width.
    Reloads .env file to reflect changes during runtime.

    Args:
        vars_per_line (int): Number of variables to display per line (default: 3).
        total_line_width (int): Total width of each line for centering (default: 120).

    Raises:
        ValueError: If vars_per_line or total_line_width is invalid.
    """
    try:
        # Input validation
        if not isinstance(vars_per_line, int) or vars_per_line <= 0:
            raise ValueError(f"Invalid vars_per_line: {vars_per_line} (must be a positive integer)")
        if not isinstance(total_line_width, int) or total_line_width <= 0:
            raise ValueError(f"Invalid total_line_width: {total_line_width} (must be a positive integer)")

        # Reload trade variables
        trade_variables = reload_trade_variables()
        if not trade_variables:
            logger.warning("No trade variables available to print")
            return

        # Print headline, centered within total_line_width
        headline = "TRADE STRATEGY VARIABLES"
        logger.info(f"{BRIGHT_BLUE}{headline.center(total_line_width)}{RESET}")
        logger.info('')

        # Determine the maximum width needed for variable names and values
        max_name_width = max(len(name) for name, _ in trade_variables) + 4  # Add padding
        max_value_width = max(len(value) for _, value in trade_variables) + 4  # Add padding
        col_width = max(max_name_width, max_value_width)

        # Calculate the width allocated to each column based on vars_per_line
        allocated_col_width = total_line_width // max(1, vars_per_line)

        # Ensure allocated_col_width is sufficient
        if allocated_col_width < col_width:
            logger.warning(
                f"Allocated column width ({allocated_col_width}) is less than required ({col_width}). "
                f"Increasing total_line_width to {col_width * vars_per_line}."
            )
            total_line_width = col_width * vars_per_line
            allocated_col_width = total_line_width // max(1, vars_per_line)

        # Process variables in chunks based on vars_per_line
        for i in range(0, len(trade_variables), vars_per_line):
            chunk = trade_variables[i:i + vars_per_line]
            name_line = ""
            value_line = ""
            for name, value in chunk:
                # Truncate name/value if they exceed allocated_col_width to prevent overflow
                name_display = (name[:allocated_col_width-4] + "..." if len(name) > allocated_col_width-4 else name)
                value_display = (value[:allocated_col_width-4] + "..." if len(value) > allocated_col_width-4 else value)
                name_line += f"{GREEN}{name_display.center(allocated_col_width)}{RESET}"
                value_line += f"{WHITE}{value_display.center(allocated_col_width)}{RESET}"
            logger.info(name_line)
            logger.info(value_line)
            logger.info('')

        logger.debug(f"Printed {len(trade_variables)} trade variables with {vars_per_line} per line")
    except ValueError as e:
        logger.error(f"Validation error in print_trade_variables: {e}", exc_info=True)
        send_alert("Print Trade Variables Failure", f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in print_trade_variables: {e}", exc_info=True)
        send_alert("Print Trade Variables Failure", f"Unexpected error: {e}")

def send_alert(subject, message):
    """
    Sends an alert for critical errors (placeholder).

    Args:
        subject (str): The subject of the alert.
        message (str): The alert message.
    """
    logger.error(f"ALERT: {subject} - {message}")