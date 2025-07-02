import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import re
from pathlib import Path

def parse_filename_timestamp(filename):
    """Extract timestamp from filename (e.g., 20250702_061228) and convert to datetime."""
    match = re.search(r'(\d{8}_\d{6})', filename)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    return None

def analyze_parquet_file(file_path, time_window_minutes=50):
    """Analyze a single Parquet file and compute sentiment metrics."""
    try:
        df = pd.read_parquet(file_path)
        
        # Adjust timestamps to CEST (UTC+2)
        timestamp_col = 'timestamp'
        df[timestamp_col] = df[timestamp_col] + timedelta(hours=2)
        
        # Filter for the latest time window
        latest_time = df[timestamp_col].max()
        start_time = latest_time - timedelta(minutes=time_window_minutes)
        df_latest = df[df[timestamp_col] >= start_time].copy()
        
        if df_latest.empty:
            return None
        
        # Check unique symbols
        all_symbols = set(df['symbol'].unique())
        window_symbols = set(df_latest['symbol'].unique())
        missing_symbols = all_symbols - window_symbols
        
        # Calculate price change and volatility
        df_latest.loc[:, 'price_change_pct'] = (df_latest['close'] - df_latest['open']) / df_latest['open'] * 100
        df_latest.loc[:, 'volatility'] = (df_latest['high'] - df_latest['low']) / df_latest['open'] * 100
        
        # Aggregate by symbol
        df_agg = df_latest.groupby('symbol').agg({
            'price_change_pct': 'mean',
            'volume': 'sum',
            'volatility': 'mean'
        }).reset_index()
        
        # Include missing symbols
        missing_df = pd.DataFrame({
            'symbol': list(missing_symbols),
            'price_change_pct': 0.0,
            'volume': 0.0,
            'volatility': 0.0
        })
        df_agg = pd.concat([df_agg, missing_df], ignore_index=True)
        
        # Advanced Metrics
        # 1. Market Breadth Index (MBI)
        df_agg['abs_price_change'] = df_agg['price_change_pct'].abs()
        mbi = (df_agg[df_agg['price_change_pct'] > 0]['abs_price_change'].sum() /
               df_agg['abs_price_change'].sum() if df_agg['abs_price_change'].sum() > 0 else 0.5)
        
        # 2. Volume-Weighted Price Momentum (VWPM)
        df_agg['volume_weighted_change'] = df_agg['price_change_pct'] * df_agg['volume']
        vwpm = (df_agg['volume_weighted_change'].sum() /
                df_agg['volume'].sum() if df_agg['volume'].sum() > 0 else 0.0)
        
        # 3. Volatility-Adjusted Sentiment (VAS)
        df_agg['normalized_change'] = df_agg['price_change_pct'] / df_agg['volatility'].replace(0, 1)
        vas = df_agg['normalized_change'].mean()
        
        # 4. McClellan Oscillator
        minute_groups = df_latest.groupby('timestamp')
        net_advancers = []
        for minute, group in minute_groups:
            advancers = len(group[group['price_change_pct'] > 0])
            decliners = len(group[group['price_change_pct'] < 0])
            net_advancers.append(advancers - decliners)
        
        if net_advancers:
            net_advancers = pd.Series(net_advancers)
            ema_short = net_advancers.ewm(span=19, adjust=False).mean().iloc[-1]
            ema_long = net_advancers.ewm(span=39, adjust=False).mean().iloc[-1]
            mc_oscillator = ema_short - ema_long
        else:
            mc_oscillator = 0.0
        
        # 5. Cross-Sectional Momentum (CSM)
        n_coins = len(df_agg)
        top_n = int(n_coins * 0.2)
        bottom_n = int(n_coins * 0.2)
        sorted_agg = df_agg.sort_values('price_change_pct')
        top_performers = sorted_agg.tail(top_n)
        bottom_performers = sorted_agg.head(bottom_n)
        csm = top_performers['price_change_pct'].mean() - bottom_performers['price_change_pct'].mean()
        
        # Basic Metrics
        advancers = len(df_agg[df_agg['price_change_pct'] > 0])
        decliners = len(df_agg[df_agg['price_change_pct'] < 0])
        ad_ratio = advancers / decliners if decliners > 0 else float('inf') if advancers > 0 else 0.0
        bullish_volume = df_agg[df_agg['price_change_pct'] > 0]['volume'].sum()
        bearish_volume = df_agg[df_agg['price_change_pct'] < 0]['volume'].sum()
        volume_ratio = bullish_volume / bearish_volume if bearish_volume > 0 else float('inf') if bullish_volume > 0 else 0.0
        
        # Total trading volume
        total_volume = df_agg['volume'].sum()
        
        # Combined Sentiment
        sentiment_scores = [
            mbi > 0.5,
            vwpm > 0,
            vas > 0,
            mc_oscillator > 0,
            csm > 0,
            ad_ratio > 1,
            volume_ratio > 1
        ]
        bullish_count = sum(sentiment_scores)
        total_metrics = len(sentiment_scores)
        bullish_metrics = bullish_count / total_metrics
        
        # Trading Signal
        if bullish_metrics >= 0.6 and abs(vwpm) > 0.05:
            trade_signal = 'Bullish'
        elif bullish_metrics <= 0.4 and abs(vwpm) > 0.05:
            trade_signal = 'Bearish'
        else:
            trade_signal = 'Neutral'
        
        return {
            'timestamp': latest_time,
            'mbi': mbi,
            'vwpm': vwpm,
            'vas': vas,
            'mc_oscillator': mc_oscillator,
            'csm': csm,
            'ad_ratio': ad_ratio,
            'volume_ratio': volume_ratio,
            'bullish_metrics': bullish_metrics,
            'trade_signal': trade_signal,
            'total_volume': total_volume,
            'n_coins': len(df_agg),
            'n_missing': len(missing_symbols)
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def visualize_market_trends_by_weekday(folder_path, time_window_minutes=50):
    """
    Aggregate Parquet files and visualize market activity and sentiment by weekday and hour.
    
    Parameters:
    - folder_path (str): Path to the folder containing Parquet files.
    - time_window_minutes (int): Number of minutes to analyze per file (default: 50).
    """
    # Get list of Parquet files
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not files:
        print("No Parquet files found in the folder.")
        return
    
    # Parse timestamps and sort files
    file_data = []
    for f in files:
        timestamp = parse_filename_timestamp(f)
        if timestamp:
            file_data.append((os.path.join(folder_path, f), timestamp))
    
    file_data.sort(key=lambda x: x[1])  # Sort by timestamp
    
    # Process each file
    results = []
    for file_path, file_timestamp in file_data:
        print(f"Processing {file_path}...")
        result = analyze_parquet_file(file_path, time_window_minutes)
        if result:
            # Extract weekday and hour from adjusted timestamp
            timestamp = result['timestamp']
            result['weekday'] = timestamp.strftime('%A')
            result['hour'] = timestamp.hour
            results.append(result)
    
    if not results:
        print("No valid data processed from files.")
        return
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Aggregate by weekday and hour
    agg_results = df_results.groupby(['weekday', 'hour']).agg({
        'bullish_metrics': 'mean',
        'total_volume': 'mean',
        'trade_signal': lambda x: pd.Series(x).mode()[0] if not x.empty else 'Neutral',
        'n_coins': 'mean',
        'n_missing': 'mean'
    }).reset_index()
    
    # Ensure all weekdays and hours are present
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    all_combinations = pd.DataFrame([(w, h) for w in weekdays for h in hours], columns=['weekday', 'hour'])
    agg_results = all_combinations.merge(agg_results, on=['weekday', 'hour'], how='left')
    agg_results['bullish_metrics'] = agg_results['bullish_metrics'].fillna(0.5)  # Neutral sentiment
    agg_results['total_volume'] = agg_results['total_volume'].fillna(0.0)  # No volume
    agg_results['trade_signal'] = agg_results['trade_signal'].fillna('Neutral')
    
    # Pivot for heatmaps
    pivot_sentiment = agg_results.pivot(index='weekday', columns='hour', values='bullish_metrics')
    pivot_volume = agg_results.pivot(index='weekday', columns='hour', values='total_volume')
    pivot_signals = agg_results.pivot(index='weekday', columns='hour', values='trade_signal')
    
    # Reorder weekdays
    pivot_sentiment = pivot_sentiment.reindex(weekdays)
    pivot_volume = pivot_volume.reindex(weekdays)
    pivot_signals = pivot_signals.reindex(weekdays)
    
    # Plot heatmaps
    plt.figure(figsize=(20, 10))
    
    # Sentiment Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(pivot_sentiment, cmap='RdYlGn', vmin=0.0, vmax=1.0, annot=True, fmt='.2f')
    plt.title('Average Bullish Sentiment by Weekday and Hour (CEST)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekday')
    
    # Volume Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(pivot_volume, cmap='Blues', annot=True, fmt='.2e')
    plt.title('Average Trading Volume by Weekday and Hour (CEST)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekday')
    
    plt.tight_layout()
    plt.savefig('market_activity_sentiment_heatmap.png')
    plt.close()
    print("Heatmaps saved as 'market_activity_sentiment_heatmap.png'")
    
    # Plot trading signals
    plt.figure(figsize=(15, 5))
    signal_values = {'Bullish': 1, 'Neutral': 0, 'Bearish': -1}
    pivot_signals_numeric = pivot_signals.replace(signal_values)
    sns.heatmap(pivot_signals_numeric, cmap='RdYlGn', vmin=-1, vmax=1, annot=pivot_signals, fmt='')
    plt.title('Dominant Trading Signal by Weekday and Hour (CEST)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Weekday')
    plt.savefig('trading_signals_heatmap.png')
    plt.close()
    print("Trading signals heatmap saved as 'trading_signals_heatmap.png'")
    
    # Save results to CSV
    agg_results.to_csv('market_trend_weekly_results.csv', index=False)
    print("Aggregated results saved to 'market_trend_weekly_results.csv'")
    
    # Identify best trading times
    best_times = agg_results[
        ((agg_results['bullish_metrics'] >= 0.6) | (agg_results['bullish_metrics'] <= 0.4)) &
        (agg_results['total_volume'] > agg_results['total_volume'].quantile(0.75))
    ][['weekday', 'hour', 'bullish_metrics', 'total_volume', 'trade_signal']]
    print("\nBest Trading Times (High Sentiment and Volume):")
    print(best_times)
    best_times.to_csv('best_trading_times.csv', index=False)
    print("Best trading times saved to 'best_trading_times.csv'")

# Example usage
folder_path = r"F:\Crypto_Trading\Market_Data"  # Replace with your folder path
visualize_market_trends_by_weekday(folder_path, time_window_minutes=50)