import json
import logging
from typing import Dict, Any
from src.config import logger

# Define ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"  # Resets color to default

def colorize_value(value: Any, column: str, current_price: float, purchase_price: float, 
                  profit_target: float) -> str:
    if column == "symbol":
        return f"{YELLOW}{value:<12}{RESET}"
    elif column == "current_price":
        return f"{GREEN if current_price > purchase_price else RED}{value:>15.8f}{RESET}"
    elif column in ["original_profit_target", "sell_price"]:
        target_price = purchase_price * (1 + profit_target)
        if current_price > target_price:
            return (f"{GREEN}{value:>6.3f}{RESET}%" if column == "original_profit_target" 
                    else f"{GREEN}{value:>15.8f}{RESET}")
        return f"{value:>6.3f}%" if column == "original_profit_target" else f"{value:>15.8f}"
    elif column == "profit_loss_percent":
        return f"{GREEN if value > 0 else RED}{value:>6.3f}%{RESET}"
    elif column == "total_value":
        return f"{GREEN if current_price > purchase_price else RED}{value:>12.2f}€{RESET}"
    return str(value)

def print_portfolio(file_path: str) -> None:
    """
    Reads a JSON portfolio file and prints its contents with color formatting.
    
    Args:
        file_path (str): Path to the JSON portfolio file
    """

    try:
        # Read JSON file
        with open(file_path, 'r') as file:
            portfolio = json.load(file)


        # Process each asset
        for symbol, data in portfolio['assets'].items():
            # Calculate additional metrics
            profit_loss_percent = ((data['current_price'] - data['purchase_price']) / 
                                 data['purchase_price'] * 100)
            total_value = data['quantity'] * data['current_price']

            # Prepare row data
            row = {
                'symbol': symbol,
                'quantity': data['quantity'],
                'purchase_price': data['purchase_price'],
                'current_price': data['current_price'],
                'profit_loss_percent': profit_loss_percent,
                'total_value': total_value,
                'original_profit_target': data['original_profit_target'] * 100,  # Convert to percentage
                'sell_price': data['sell_price'],
                'purchase_time': data['purchase_time']
            }

            
            # Print formatted row
            logger.info(
                f"Symbol: {colorize_value(row['symbol'], 'symbol', row['current_price'], row['purchase_price'], row['original_profit_target'])}  "
                f"Quantity: {row['quantity']:>12.2f}  "
                f"Purchase Price: {row['purchase_price']:>15.8f}  "
                f"Current Price: {colorize_value(row['current_price'], 'current_price', row['current_price'], row['purchase_price'], row['original_profit_target'])}  "
                f"P/L %: {colorize_value(row['profit_loss_percent'], 'profit_loss_percent', row['current_price'], row['purchase_price'], row['original_profit_target'])}  "
                f"Total Value: {colorize_value(row['total_value'], 'total_value', row['current_price'], row['purchase_price'], row['original_profit_target'])}  "
                f"Profit Target: {colorize_value(row['original_profit_target'], 'original_profit_target', row['current_price'], row['purchase_price'], row['original_profit_target'])}  "
                f"Sell Price: {colorize_value(row['sell_price'], 'sell_price', row['current_price'], row['purchase_price'], row['original_profit_target'])}  "
                f"Purchase Time: {row['purchase_time']}"
            )

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {file_path}")
    except Exception as e:
        logger.error(f"Error processing portfolio: {str(e)}")
