import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import asyncio
from scipy.stats import norm

class EarningsStraddle:
    def __init__(self):
        self.cache_timeout = 300  # 5 minutes cache timeout
        self.last_update = None
        self.cached_data = None
        self.risk_free_rate = 0.05  # 5% risk-free rate

    def get_large_caps_with_earnings(self):
        """Get large cap stocks with upcoming earnings"""
        large_caps = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'ORCL', 
            'ACN', 'INTC', 'AMD', 'IBM', 'NOW', 'QCOM', 'INFY', 'MU', 'ADI', 'AMAT', 'LRCX',
            
            # Communication Services
            'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'MAR',
            
            # Consumer Staples
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'TGT', 'EL', 'CL', 'KHC',
            
            # Financials
            'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'SCHW', 'AXP', 'C',
            'SPGI', 'CB', 'MMC', 'PGR', 'TFC',
            
            # Healthcare
            'JNJ', 'UNH', 'LLY', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'CVS',
            'MRK', 'MDT', 'GILD', 'ISRG', 'REGN', 'VRTX', 'ZTS',
            
            # Industrials
            'RTX', 'HON', 'UPS', 'BA', 'CAT', 'LMT', 'GE', 'MMM', 'DE', 'ADP',
            'GD', 'EMR', 'ETN', 'NSC', 'ITW',
            
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'PSX',
            
            # Materials
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW',
            
            # Real Estate
            'PLD', 'AMT', 'EQIX', 'CCI', 'SPG',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'AEP', 'EXC'
        ]
        
        # Modify the delay based on the number of stocks processed
        request_delay = 0.2  # Reduced delay to 0.2 seconds between requests
        
        stocks_with_earnings = []
        total_stocks = len(large_caps)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(large_caps):
            try:
                # Update progress
                progress = (idx + 1) / total_stocks
                progress_bar.progress(progress)
                status_text.text(f"Processing {symbol} ({idx + 1}/{total_stocks})")
                
                stock = yf.Ticker(symbol)
                next_earnings = stock.calendar
                
                if next_earnings is not None and isinstance(next_earnings, dict):
                    earnings_date_raw = next_earnings.get('Earnings Date', None)
                    
                    # Handle different possible formats of earnings_date
                    if isinstance(earnings_date_raw, list) and len(earnings_date_raw) > 0:
                        earnings_date = pd.Timestamp(earnings_date_raw[0])
                    elif earnings_date_raw is not None:
                        earnings_date = pd.Timestamp(earnings_date_raw)
                    else:
                        earnings_date = None
                    
                    # Get the latest price data
                    try:
                        current_price = stock.fast_info['lastPrice']
                        market_cap = stock.fast_info['marketCap']
                    except:
                        # Fallback method if fast_info fails
                        hist = stock.history(period='1d')
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            market_cap = current_price * stock.fast_info['shares']
                        else:
                            continue
                    
                    # Only include if market cap > $10B and earnings within next 30 days
                    if (earnings_date is not None and 
                        market_cap > 1e10 and 
                        earnings_date - pd.Timestamp.now() < pd.Timedelta(days=30)):
                        
                        options_data = self.get_options_data(stock, current_price, earnings_date)
                        if options_data:
                            stocks_with_earnings.append({
                                'symbol': symbol,
                                'earnings_date': earnings_date,
                                'current_price': current_price,
                                'market_cap': market_cap,
                                **options_data
                            })
                
                time.sleep(request_delay)
                
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Clear the progress bar and status text
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(stocks_with_earnings)

    def get_options_data(self, stock, current_price, earnings_date):
        """Get real options data for the stock"""
        try:
            # Get options expirations
            expirations = stock.options
            
            if not expirations:
                return None
                
            # Find the expiration date closest to but after earnings
            valid_expirations = [exp for exp in expirations 
                               if pd.to_datetime(exp) > earnings_date]
            
            if not valid_expirations:
                return None
                
            nearest_expiry = min(valid_expirations)
            
            # Get options chain
            chain = stock.option_chain(nearest_expiry)
            
            # Find ATM options
            calls = chain.calls
            puts = chain.puts
            
            # Find the strike price closest to current price
            calls['strike_diff'] = abs(calls['strike'] - current_price)
            puts['strike_diff'] = abs(puts['strike'] - current_price)
            
            atm_call = calls.loc[calls['strike_diff'].idxmin()]
            atm_put = puts.loc[puts['strike_diff'].idxmin()]
            
            # Calculate implied volatility (average of call and put IV)
            avg_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2
            
            return {
                'expiration': nearest_expiry,
                'strike': atm_call['strike'],
                'call_bid': atm_call['bid'],
                'call_ask': atm_call['ask'],
                'put_bid': atm_put['bid'],
                'put_ask': atm_put['ask'],
                'call_volume': atm_call['volume'],
                'put_volume': atm_put['volume'],
                'call_iv': atm_call['impliedVolatility'],
                'put_iv': atm_put['impliedVolatility'],
                'avg_iv': avg_iv,
                'straddle_cost': atm_call['ask'] + atm_put['ask'],
                'call_delta': atm_call['delta'] if 'delta' in atm_call else None,
                'put_delta': atm_put['delta'] if 'delta' in atm_put else None,
                'call_theta': atm_call['theta'] if 'theta' in atm_call else None,
                'put_theta': atm_put['theta'] if 'theta' in atm_put else None,
                'call_gamma': atm_call['gamma'] if 'gamma' in atm_call else None,
                'put_gamma': atm_put['gamma'] if 'gamma' in atm_put else None,
            }
            
        except Exception as e:
            st.error(f"Error getting options data: {str(e)}")
            return None

    def get_data(self):
        """Get data with caching"""
        current_time = time.time()
        
        # Return cached data if it's still valid
        if (self.last_update is not None and 
            current_time - self.last_update < self.cache_timeout and 
            self.cached_data is not None):
            return self.cached_data
            
        # Get fresh data
        self.cached_data = self.get_large_caps_with_earnings()
        
        # Filter for high probability trades
        if not self.cached_data.empty:
            self.cached_data = self.filter_high_probability_trades(self.cached_data)
            
        self.last_update = current_time
        return self.cached_data

    def filter_high_probability_trades(self, df):
        """Filter for high probability trades based on various criteria"""
        # Calculate days until earnings
        df['days_to_earnings'] = (pd.to_datetime(df['earnings_date']) - pd.Timestamp.now()).dt.days
        
        # Calculate straddle cost as percentage of stock price
        df['straddle_percentage'] = (df['straddle_cost'] / df['current_price']) * 100
        
        # Calculate option liquidity score (based on volume)
        df['total_volume'] = df['call_volume'] + df['put_volume']
        
        # Filter criteria
        filtered_df = df[
            # Adequate liquidity
            (df['total_volume'] > 100) &
            
            # Reasonable implied volatility (not too high or low)
            (df['avg_iv'] > 0.2) & 
            (df['avg_iv'] < 1.5) &
            
            # Earnings within next 2 weeks
            (df['days_to_earnings'] <= 14) &
            (df['days_to_earnings'] > 0) &
            
            # Reasonable straddle cost (not too expensive relative to stock price)
            (df['straddle_percentage'] < 10) &
            
            # Ensure bid-ask spreads aren't too wide
            ((df['call_ask'] - df['call_bid']) / df['call_ask'] < 0.15) &
            ((df['put_ask'] - df['put_bid']) / df['put_ask'] < 0.15)
        ]
        
        # Calculate a composite score for ranking
        filtered_df['probability_score'] = (
            filtered_df['total_volume'] / filtered_df['total_volume'].max() * 0.3 +  # Volume weight
            (1 - filtered_df['straddle_percentage'] / filtered_df['straddle_percentage'].max()) * 0.3 +  # Cost weight
            (1 - filtered_df['days_to_earnings'] / filtered_df['days_to_earnings'].max()) * 0.2 +  # Time weight
            filtered_df['avg_iv'] / filtered_df['avg_iv'].max() * 0.2  # IV weight
        )
        
        # Sort by probability score and take top results
        return filtered_df.sort_values('probability_score', ascending=False).head(10)

def main():
    st.title("Live Earnings Straddle Analysis")
    
    # Move session state initialization to the top
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    st.write("""
    This tool analyzes potential straddle opportunities for upcoming earnings announcements using real-time options data.
    Data refreshes every 5 minutes to respect API limits.
    """)
    
    # Initialize analyzer
    analyzer = EarningsStraddle()
    
    # Add auto-refresh
    auto_refresh = st.checkbox('Enable auto-refresh (5 min interval)', value=True)
    
    current_time = time.time()
    if auto_refresh:
        # Calculate time since last refresh
        time_since_refresh = current_time - st.session_state.last_refresh
        time_until_next = max(300 - time_since_refresh, 0)  # 300 seconds = 5 minutes
        
        # Display countdown
        st.write(f"Next refresh in: {int(time_until_next)} seconds")
        st.write(f"Last update: {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Only refresh if 5 minutes have passed
        if time_since_refresh >= 300:
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Get data
    results_df = analyzer.get_data()
    
    if results_df.empty:
        st.warning("No suitable earnings plays found at the moment.")
        return
    
    # Display results
    st.subheader("Top 10 Highest Probability Earnings Straddle Opportunities")
    
    # Format the display DataFrame
    display_df = results_df[[
        'symbol', 'earnings_date', 'current_price', 'strike',
        'straddle_cost', 'straddle_percentage', 'avg_iv', 
        'total_volume', 'days_to_earnings', 'probability_score'
    ]].copy()
    
    # Format the columns
    display_df['earnings_date'] = pd.to_datetime(display_df['earnings_date']).dt.strftime('%Y-%m-%d')
    display_df['avg_iv'] = (display_df['avg_iv'] * 100).round(1).astype(str) + '%'
    display_df['probability_score'] = display_df['probability_score'].round(3)
    display_df['straddle_percentage'] = display_df['straddle_percentage'].round(1).astype(str) + '%'
    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
    display_df['strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
    display_df['straddle_cost'] = display_df['straddle_cost'].apply(lambda x: f"${x:.2f}")
    
    # Sort by probability score
    display_df = display_df.sort_values('probability_score', ascending=False)
    
    # Rename columns for better display
    display_df.columns = [
        'Symbol', 'Earnings Date', 'Current Price', 'Strike',
        'Straddle Cost', 'Cost %', 'IV %', 
        'Total Volume', 'Days to Earnings', 'Probability Score'
    ]
    
    # Display the dataframe with highlighting using native Streamlit features
    st.dataframe(
        display_df,
        column_config={
            "Probability Score": st.column_config.NumberColumn(
                format="%.3f",
                help="Higher scores indicate better opportunities"
            ),
            "Total Volume": st.column_config.NumberColumn(
                format="%d",
                help="Combined call and put volume"
            ),
            "Days to Earnings": st.column_config.NumberColumn(
                format="%d",
                help="Days until earnings announcement"
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Add explanation of scoring
    st.info("""
    **Probability Score Factors:**
    - Higher trading volume (better liquidity)
    - Lower straddle cost relative to stock price
    - Closer to earnings date
    - Moderate implied volatility
    - Tight bid-ask spreads
    
    Scores range from 0 to 1, with higher scores indicating potentially better opportunities.
    """)
    
    # Detailed analysis for each stock
    st.subheader("Detailed Analysis by Stock")
    for _, row in results_df.iterrows():
        with st.expander(f"Show {row['symbol']} Details"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Option Prices**")
                st.write(f"Strike: ${row['strike']:.2f}")
                st.write(f"Call Bid/Ask: ${row['call_bid']:.2f}/${row['call_ask']:.2f}")
                st.write(f"Put Bid/Ask: ${row['put_bid']:.2f}/${row['put_ask']:.2f}")
                st.write(f"Total Straddle Cost: ${row['straddle_cost']:.2f}")
            
            with col2:
                st.write("**Greeks**")
                if row['call_delta'] is not None:
                    st.write(f"Call Delta: {row['call_delta']:.3f}")
                    st.write(f"Put Delta: {row['put_delta']:.3f}")
                    st.write(f"Gamma: {row['call_gamma']:.6f}")
                    st.write(f"Call Theta: ${row['call_theta']:.2f}")
                    st.write(f"Put Theta: ${row['put_theta']:.2f}")
                else:
                    st.write("Greeks not available")
            
            with col3:
                st.write("**Volume & Volatility**")
                st.write(f"Call Volume: {row['call_volume']}")
                st.write(f"Put Volume: {row['put_volume']}")
                st.write(f"Call IV: {row['call_iv']*100:.1f}%")
                st.write(f"Put IV: {row['put_iv']*100:.1f}%")
                st.write(f"Average IV: {row['avg_iv']*100:.1f}%")

if __name__ == "__main__":
    main()
