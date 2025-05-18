import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import random

__author__= "Carolina Kogan, Alice Node Langlois, Ismail Guennoun and Jan Budzisz"
# Set page config
st.set_page_config(
    page_title="Advanced Stock Market Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Apply custom CSS
def load_css():
    with open("final/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Additional CSS directly in the code for immediate styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #0f2537 !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    .sub-header {
        font-size: 14px !important;
        color: #4c566a !important;
        font-style: italic !important;
        margin-top: 0px !important;
    }
    .metric-card {
        background-color: #fff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    .metric-label {
        font-size: 14px !important;
        color: #4c566a !important;
    }
    .positive {
        color: #00ab41 !important;
    }
    .negative {
        color: #e31937 !important;
    }
    .divider {
        height: 1px;
        background-color: #eaeaea;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)


try:
    load_css()
except:
    st.warning("Custom CSS file not found. Using default styling.")

# Header with modern styling
st.title("**Stock Market Dashboard**")
st.markdown('<p class="sub-header">Professional market analysis and performance tracking</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">By group 4: Carolina Kogan, Ismail Guennoun, Jan Budzsiz and Alice Nodé-Langlois</p>', unsafe_allow_html=True)
# Create a professional sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 20px;">
    <h3 style="color: #0f2537;">Dashboard Controls</h3>
</div>
""", unsafe_allow_html=True)

# Stock selector and date range
st.sidebar.markdown("### Stock Selection")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker Symbol", "AAPL").upper().strip()

# Date inputs with more professional styling
st.sidebar.markdown("### Time Period")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", date.today() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", date.today())

# Display options with better organization
st.sidebar.markdown("### Chart Settings")
with st.sidebar.expander("Display Options", expanded=True):
    y_scale = st.radio(
        "Y-Axis Scale",
        ["Linear", "Logarithmic"],
        index=0
    )

    time_frame = st.radio(
        "Time Granularity",
        ["Daily", "Weekly", "Monthly"],
        index=0
    )

    show_volume = st.checkbox("Show Volume Chart", value=True)

    metrics_display = st.radio(
        "Metrics Display Style",
        ["Cards", "Table", "Both"],
        index=0  # Changed default to Cards for a more modern look
    )

    color_scheme = st.radio(
        "Color Scheme",
        ["Green/Red (Traditional)", "Blue/Orange (Colorblind-friendly)"],
        index=0
    )

# Input validation with improved styling
if start_date > end_date:
    st.sidebar.error("⚠️ Start date must be before end date")
    st.stop()

if start_date > date.today():
    st.sidebar.error("⚠️ Start date must be before today")
    st.stop()

if end_date > date.today():
    st.sidebar.error("⚠️ End date must be before today")
    st.stop()


# Cache the sample data generation to improve performance
@st.cache_data(ttl=3600)
def generate_sample_data(ticker, start_date, end_date, time_frame="Daily"):
    """Generate realistic sample stock data for testing"""

    # Create date range based on the selected time frame
    if time_frame == "Daily":
        # For daily data, include business days only
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    elif time_frame == "Weekly":
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    elif time_frame == "Monthly":
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Set seed based on ticker for consistent results
    seed = sum(ord(c) for c in ticker)
    np.random.seed(seed)

    # Different tickers have different price ranges and volatility
    if ticker == "AAPL":
        base_price = 150
        volatility = 0.015
        trend = 0.0002
    elif ticker == "MSFT":
        base_price = 280
        volatility = 0.012
        trend = 0.0003
    elif ticker == "GOOGL":
        base_price = 120
        volatility = 0.018
        trend = 0.0001
    elif ticker == "AMZN":
        base_price = 100
        volatility = 0.02
        trend = 0.0002
    else:
        # Random values for other tickers
        base_price = np.random.randint(50, 500)
        volatility = np.random.uniform(0.01, 0.03)
        trend = np.random.uniform(0.0001, 0.0005) * (1 if np.random.random() > 0.3 else -1)

    # Generate price series with random walk and trend
    n = len(date_range)

    # Create some market events/volatility periods
    events = []
    for _ in range(3):
        event_day = np.random.randint(0, n)
        event_impact = np.random.choice([-1, 1]) * np.random.uniform(0.03, 0.08)
        event_length = np.random.randint(5, 15)
        events.append((event_day, event_impact, event_length))

    # Generate daily returns with events
    daily_returns = np.random.normal(trend, volatility, n)

    # Add events to returns
    for event_day, event_impact, event_length in events:
        if event_day < n:
            daily_returns[event_day] += event_impact
            # Aftershocks
            for i in range(1, event_length):
                if event_day + i < n:
                    daily_returns[event_day + i] += event_impact * (event_length - i) / event_length * 0.5

    # Convert returns to price series
    prices = [base_price]
    for ret in daily_returns:
        prices.append(prices[-1] * (1 + ret))
    prices = prices[1:]  # Remove the seed price

    # Create realistic OHLC data
    opens = prices.copy()
    closes = []
    highs = []
    lows = []

    # Adjust opens to be previous day's close except for first day
    opens = [prices[0]] + prices[:-1]

    for i, price in enumerate(prices):
        # Daily close is the main price we generated
        close = price
        opens[i] = opens[i] * (1 + np.random.normal(0, 0.002))  # Slight adjustment to open

        # High is above both open and close
        high = max(opens[i], close) * (1 + abs(np.random.normal(0, 0.005)))

        # Low is below both open and close
        low = min(opens[i], close) * (1 - abs(np.random.normal(0, 0.005)))

        closes.append(close)
        highs.append(high)
        lows.append(low)

    # Generate volume data - higher on big move days
    volume_base = np.random.randint(1000000, 10000000)
    volumes = []

    for ret in daily_returns:
        # Higher volume on days with bigger price moves
        vol_multiplier = 1 + 5 * abs(ret - trend) / volatility
        daily_vol = int(volume_base * vol_multiplier * np.random.uniform(0.8, 1.2))
        volumes.append(daily_vol)

    # Create DataFrame
    sample_data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=date_range)

    # Resample if needed
    if time_frame == "Weekly" and len(sample_data) > 7:
        sample_data = sample_data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif time_frame == "Monthly" and len(sample_data) > 30:
        sample_data = sample_data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

    return sample_data


# Generate sample data
with st.spinner(f"Generating sample data for {ticker_symbol}..."):
    stock_data = generate_sample_data(ticker_symbol, start_date, end_date, time_frame)


# Calculate key performance metrics
def calculate_metrics(stock_data):
    """Calculate key financial metrics from stock data"""
    metrics = {}

    # Calculate daily returns
    stock_data['Daily Return'] = stock_data['Close'].pct_change() * 100

    # Return metrics
    metrics['Total Return (%)'] = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
    # Annualized return (252 trading days in a year)
    trading_days_per_year = 252
    days_in_period = (stock_data.index[-1] - stock_data.index[0]).days
    years = days_in_period / 365.25
    metrics['Annualized Return (%)'] = ((1 + metrics['Total Return (%)'] / 100) ** (
            1 / years) - 1) * 100 if years > 0 else 0
    metrics['Average Daily Return (%)'] = stock_data['Daily Return'].mean()

    # Risk metrics
    metrics['Volatility (Daily %)'] = stock_data['Daily Return'].std()
    metrics['Annualized Volatility (%)'] = metrics['Volatility (Daily %)'] * np.sqrt(trading_days_per_year)

    # Calculate drawdown
    stock_data['Cumulative Return'] = (1 + stock_data['Daily Return'] / 100).cumprod()
    stock_data['Running Max'] = stock_data['Cumulative Return'].cummax()
    stock_data['Drawdown'] = (stock_data['Cumulative Return'] / stock_data['Running Max'] - 1) * 100
    metrics['Maximum Drawdown (%)'] = stock_data['Drawdown'].min()

    # Value at Risk (95%)
    metrics['Value at Risk (95%)'] = np.percentile(stock_data['Daily Return'], 5)

    # Trading metrics
    metrics['Positive Days (%)'] = (stock_data['Daily Return'] > 0).mean() * 100
    metrics['Negative Days (%)'] = (stock_data['Daily Return'] < 0).mean() * 100
    metrics['Average Volume'] = stock_data['Volume'].mean()

    # Calculate Sharpe Ratio (assuming risk-free rate of 0% for simplicity)
    metrics['Sharpe Ratio'] = (metrics['Average Daily Return (%)'] / metrics['Volatility (Daily %)']) * np.sqrt(
        trading_days_per_year) if metrics['Volatility (Daily %)'] > 0 else 0

    return metrics, stock_data


# Calculate metrics
metrics, stock_data_with_metrics = calculate_metrics(stock_data.copy())

# Reset index to make Date a column for the UI
stock_data_reset = stock_data.reset_index()
stock_data_with_metrics_reset = stock_data_with_metrics.reset_index()

# Create dashboard layout
# Use columns for a more professional dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    # Current price and summary section
    latest_price = stock_data['Close'].iloc[-1]
    previous_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else stock_data['Close'].iloc[0]
    price_change = latest_price - previous_price
    price_change_pct = (price_change / previous_price) * 100

    # Summary card
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin-bottom: 0;">{ticker_symbol}</h2>
                <p style="color: #6c757d; margin-top: 0;">{'Up to date as of ' + stock_data.index[-1].strftime('%b %d, %Y')}</p>
            </div>
            <div style="text-align: right;">
                <h2 style="margin-bottom: 0;">${latest_price:.2f}</h2>
                <p style="margin-top: 0; color: {'#00ab41' if price_change >= 0 else '#e31937'};">
                    {price_change:+.2f} ({price_change_pct:+.2f}%)
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stock price chart with improved styling
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_data_reset['index'],
            open=stock_data_reset['Open'],
            high=stock_data_reset['High'],
            low=stock_data_reset['Low'],
            close=stock_data_reset['Close'],
            name='Price',
            increasing_line_color='#00ab41',
            decreasing_line_color='#e31937'
        )
    )

    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=stock_data_reset['index'],
            y=stock_data_reset['Close'].rolling(window=20).mean(),
            mode='lines',
            name='20-Day MA',
            line=dict(color='#2962ff', width=1.5)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stock_data_reset['index'],
            y=stock_data_reset['Close'].rolling(window=50).mean(),
            mode='lines',
            name='50-Day MA',
            line=dict(color='#ff6d00', width=1.5)
        )
    )

    # Improve layout
    fig.update_layout(
        title=None,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        height=450,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
    )

    # Apply log scale if selected
    if y_scale == "Logarithmic":
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)

    # Add volume chart if requested
    if show_volume:
        colors = []
        for i in range(len(stock_data_reset)):
            if i > 0 and stock_data_reset['Close'].iloc[i] > stock_data_reset['Close'].iloc[i - 1]:
                colors.append('#00ab41')  # Green for up days
            else:
                colors.append('#e31937')  # Red for down days

        vol_fig = go.Figure()
        vol_fig.add_trace(
            go.Bar(
                x=stock_data_reset['index'],
                y=stock_data_reset['Volume'],
                marker_color=colors,
                name='Volume'
            )
        )

        vol_fig.update_layout(
            title='Trading Volume',
            title_font=dict(size=16),
            height=200,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title=None,
            yaxis_title="Volume",
            template="plotly_white",
        )

        # Format y-axis with K, M suffixes
        vol_fig.update_yaxes(
            tickformat=".2s",
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.6)'
        )

        st.plotly_chart(vol_fig, use_container_width=True)

with col2:
    # Key metrics in cards
    st.markdown("### Key Metrics")

    # Current price and change
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
        <h4 style="margin-top: 0;">Current Price</h4>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 24px; font-weight: bold;">${latest_price:.2f}</span>
            <span style="font-size: 16px; color: {'#00ab41' if price_change >= 0 else '#e31937'};">
                {price_change:+.2f} ({price_change_pct:+.2f}%)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Total return
    total_return = metrics['Total Return (%)']
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
        <h4 style="margin-top: 0;">Total Return</h4>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 24px; font-weight: bold;">{total_return:.2f}%</span>
            <span style="font-size: 16px; color: {'#00ab41' if total_return >= 0 else '#e31937'};">
                {total_return:+.2f}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Volatility card
    volatility = metrics['Annualized Volatility (%)']
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
        <h4 style="margin-top: 0;">Annualized Volatility</h4>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 24px; font-weight: bold;">{volatility:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sharpe ratio card
    sharpe = metrics['Sharpe Ratio']
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
        <h4 style="margin-top: 0;">Sharpe Ratio</h4>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 24px; font-weight: bold;">{sharpe:.2f}</span>
            <span style="font-size: 16px; color: {'#00ab41' if sharpe >= 1 else '#6c757d'};">
                {'Good' if sharpe >= 1 else 'Poor'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Max drawdown card
    drawdown = metrics['Maximum Drawdown (%)']
    st.markdown(f"""
    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
        <h4 style="margin-top: 0;">Maximum Drawdown</h4>
        <div style="display: flex; justify-content: space-between;">
            <span style="font-size: 24px; font-weight: bold;">{drawdown:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add a divider
st.markdown('<hr style="margin: 30px 0; border: 0; height: 1px; background: #eaeaea;">', unsafe_allow_html=True)

# Advanced metrics section
st.markdown("## Performance Analysis")

tab1, tab2, tab3 = st.tabs(["📊 All Metrics", "📈 Returns Distribution", "📋 Historical Data"])

with tab1:
    # Organize metrics by category with improved styling
    if metrics_display in ["Cards", "Both"]:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div style="text-align: center;"><h3>Return Metrics</h3></div>', unsafe_allow_html=True)
            for name, value in {
                "Total Return": f"{metrics['Total Return (%)']:.2f}%",
                "Annualized Return": f"{metrics['Annualized Return (%)']:.2f}%",
                "Avg Daily Return": f"{metrics['Average Daily Return (%)']:.2f}%"
            }.items():
                is_positive = float(value.replace('%', '')) >= 0
                color_class = "positive" if is_positive else "negative"
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin-bottom: 0px; font-size: 14px; color: #6c757d;">{name}</p>
                    <p style="font-size: 20px; font-weight: bold; margin-top: 0px; color: {'#00ab41' if is_positive else '#e31937'};">
                        {value}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown('<div style="text-align: center;"><h3>Risk Metrics</h3></div>', unsafe_allow_html=True)
            for name, value in {
                "Daily Volatility": f"{metrics['Volatility (Daily %)']:.2f}%",
                "Annual Volatility": f"{metrics['Annualized Volatility (%)']:.2f}%",
                "Maximum Drawdown": f"{metrics['Maximum Drawdown (%)']:.2f}%"
            }.items():
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin-bottom: 0px; font-size: 14px; color: #6c757d;">{name}</p>
                    <p style="font-size: 20px; font-weight: bold; margin-top: 0px;">
                        {value}
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col3:
            st.markdown('<div style="text-align: center;"><h3>Trading Metrics</h3></div>', unsafe_allow_html=True)
            for name, value in {
                "Value at Risk (95%)": f"{metrics['Value at Risk (95%)']:.2f}%",
                "Positive Days": f"{metrics['Positive Days (%)']:.2f}%",
                "Sharpe Ratio": f"{metrics['Sharpe Ratio']:.2f}"
            }.items():
                is_positive = (name == "Sharpe Ratio" and float(value) > 1) or \
                              (name == "Positive Days" and float(value.replace('%', '')) > 50)
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <p style="margin-bottom: 0px; font-size: 14px; color: #6c757d;">{name}</p>
                    <p style="font-size: 20px; font-weight: bold; margin-top: 0px; color: {'#00ab41' if is_positive else '#212529'};">
                        {value}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # Display metrics as table with improved styling
    if metrics_display in ["Table", "Both"]:
        # Create a complete metrics table
        metrics_table = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [format_metric(v) for v in metrics.values()],
            'Category': ['Return' if 'Return' in k else 'Risk' if any(
                word in k for word in ['Volatility', 'Drawdown', 'Risk', 'Sharpe']) else 'Trading' for k in
                         metrics.keys()]
        })

        # Display the table with improved styling
        st.dataframe(
            metrics_table,
            use_container_width=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),
                "Value": st.column_config.TextColumn("Value"),
                "Category": st.column_config.TextColumn("Category"),
            },
            hide_index=True
        )

with tab2:
    # Create a more professional returns distribution visualization
    if 'Daily Return' in stock_data_with_metrics_reset.columns:
        # Create side-by-side histograms for better visualization
        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig_hist = px.histogram(
                stock_data_with_metrics_reset,
                x='Daily Return',
                nbins=50,
                title="Daily Returns Distribution",
                labels={'Daily Return': 'Daily Return (%)'},
                template="plotly_white",
                color_discrete_sequence=['#0f2537']
            )

            # Add a vertical line at zero
            fig_hist.add_vline(
                x=0,
                line_width=2,
                line_dash="dash",
                line_color="#ff6d00"
            )

            # Improve chart formatting
            fig_hist.update_layout(
                height=400,
                bargap=0.1,
                xaxis_title_text='Daily Return (%)',
                yaxis_title_text='Frequency',
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        with col2:
            # Add a box plot for a more complete view of the distribution
            fig_box = px.box(
                stock_data_with_metrics_reset,
                y='Daily Return',
                title="Daily Returns Distribution Statistics",
                template="plotly_white",
                points="all",
                color_discrete_sequence=['#2962ff']
            )

            # Add a horizontal line at zero
            fig_box.add_hline(
                y=0,
                line_width=2,
                line_dash="dash",
                line_color="#ff6d00"
            )

            # Improve chart formatting
            fig_box.update_layout(
                height=400,
                yaxis_title_text='Daily Return (%)',
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig_box, use_container_width=True)

        # Add some statistics
        st.markdown("### Return Statistics")

        if len(stock_data_with_metrics) > 0 and 'Daily Return' in stock_data_with_metrics.columns:
            daily_returns = stock_data_with_metrics['Daily Return'].dropna()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{daily_returns.mean():.2f}%")
            with col2:
                st.metric("Median", f"{daily_returns.median():.2f}%")
            with col3:
                st.metric("Min", f"{daily_returns.min():.2f}%")
            with col4:
                st.metric("Max", f"{daily_returns.max():.2f}%")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Std Dev", f"{daily_returns.std():.2f}%")
            with col2:
                st.metric("Skewness", f"{daily_returns.skew():.2f}")
            with col3:
                st.metric("Kurtosis", f"{daily_returns.kurtosis():.2f}")
            with col4:
                st.metric("Positive Days", f"{(daily_returns > 0).mean() * 100:.1f}%")

with tab3:
    # Create and display data table with historical data
    if 'Daily Return' in stock_data_with_metrics_reset.columns:
        # Date filter for historical data
        date_options = stock_data_with_metrics_reset['index'].dt.strftime('%Y-%m-%d').unique()
        if len(date_options) > 30:
            st.warning("Showing recent data. Use the filter to see specific dates.")

        # Add date filter
        start_filter = st.date_input("Filter from date",
                                     value=pd.to_datetime(date_options[-min(30, len(date_options))]))

        # Filter the data
        filtered_data = stock_data_with_metrics_reset[
            stock_data_with_metrics_reset['index'] >= pd.to_datetime(start_filter)]

        # Select relevant columns with improved formatting
        hist_data = filtered_data[['index', 'Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return']]
        hist_data = hist_data.rename(columns={'index': 'Date', 'Daily Return': 'Daily Return (%)'})

        # Display the data with improved styling
        st.dataframe(
            hist_data,
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Open": st.column_config.NumberColumn("Open", format="$%.2f"),
                "High": st.column_config.NumberColumn("High", format="$%.2f"),
                "Low": st.column_config.NumberColumn("Low", format="$%.2f"),
                "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
                "Volume": st.column_config.NumberColumn("Volume", format="%d"),
                "Daily Return (%)": st.column_config.NumberColumn("Daily Return (%)",
                                                                  format="%.2f%%")
            },
            height=400
        )

        # Add download button with improved styling
        st.download_button(
            label="📥 Download Data as CSV",
            data=hist_data.to_csv().encode('utf-8'),
            file_name=f"{ticker_symbol}_stock_data.csv",
            mime="text/csv",
        )

# Create a drawdown visualization
st.markdown("## Drawdown Analysis")
st.markdown("Drawdown measures the decline from a historical peak in percentage terms")

if 'Drawdown' in stock_data_with_metrics.columns:
    # Reset index
    drawdown_data = stock_data_with_metrics.reset_index()

    # Create drawdown chart
    fig_dd = px.area(
        drawdown_data,
        x='index',
        y='Drawdown',
        title="Historical Drawdowns",
        labels={'Drawdown': 'Drawdown (%)', 'index': 'Date'},
        template="plotly_white",
        color_discrete_sequence=['#e31937']
    )

    # Improve chart formatting
    fig_dd.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Invert y-axis for better visualization (drawdowns are negative)
    fig_dd.update_yaxes(autorange="reversed")

    st.plotly_chart(fig_dd, use_container_width=True)

    # Add drawdown statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maximum Drawdown", f"{stock_data_with_metrics['Drawdown'].min():.2f}%")
    with col2:
        st.metric("Average Drawdown", f"{stock_data_with_metrics['Drawdown'].mean():.2f}%")
    with col3:
        recovery_time = "N/A"
        if stock_data_with_metrics['Drawdown'].min() == stock_data_with_metrics['Drawdown'].iloc[-1]:
            recovery_time = "Ongoing"
        else:
            # This is a simplification, actual recovery time calculation would be more complex
            min_dd_idx = stock_data_with_metrics.index.get_loc(stock_data_with_metrics['Drawdown'].idxmin())
            if min_dd_idx < len(stock_data_with_metrics) - 1:
                recovery_days = len(stock_data_with_metrics) - min_dd_idx - 1
                recovery_time = f"{recovery_days} days"
        st.metric("Recovery Time", recovery_time)

# Performance comparison section
st.markdown("## Market Comparison")
st.markdown("Compare performance against market benchmarks")


# Create sample benchmark data (normally you'd fetch this from an API)
@st.cache_data(ttl=3600)
def generate_benchmark_data(start_date, end_date):
    """Generate sample benchmark data for comparison"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')

    # Generate SPY (S&P 500 ETF) data
    spy_seed = 12345
    np.random.seed(spy_seed)
    spy_base = 400  # Approximate SPY price
    spy_vol = 0.01
    spy_trend = 0.0001

    # Generate QQQ (Nasdaq ETF) data
    qqq_seed = 67890
    np.random.seed(qqq_seed)
    qqq_base = 350  # Approximate QQQ price
    qqq_vol = 0.015
    qqq_trend = 0.00015

    # Generate daily returns
    n = len(date_range)
    spy_returns = np.random.normal(spy_trend, spy_vol, n)
    qqq_returns = np.random.normal(qqq_trend, qqq_vol, n)

    # Convert to price series
    spy_prices = [spy_base]
    qqq_prices = [qqq_base]

    for i in range(n):
        spy_prices.append(spy_prices[-1] * (1 + spy_returns[i]))
        qqq_prices.append(qqq_prices[-1] * (1 + qqq_returns[i]))

    spy_prices = spy_prices[1:]
    qqq_prices = qqq_prices[1:]

    # Create DataFrame
    benchmark_data = pd.DataFrame({
        'SPY': spy_prices,
        'QQQ': qqq_prices
    }, index=date_range)

    return benchmark_data


# Get benchmark data
benchmark_data = generate_benchmark_data(start_date, end_date)

# Combine with stock data for comparison
combined_data = pd.DataFrame({
    ticker_symbol: stock_data['Close'] / stock_data['Close'].iloc[0] * 100,
    'S&P 500': benchmark_data['SPY'] / benchmark_data['SPY'].iloc[0] * 100,
    'NASDAQ': benchmark_data['QQQ'] / benchmark_data['QQQ'].iloc[0] * 100
})

# Create comparison chart
combined_data_reset = combined_data.reset_index()
fig_compare = px.line(
    combined_data_reset,
    x='index',
    y=[ticker_symbol, 'S&P 500', 'NASDAQ'],
    title="Relative Performance (Normalized to 100)",
    labels={'value': 'Relative Performance', 'index': 'Date', 'variable': 'Asset'},
    template="plotly_white"
)

# Improve chart formatting
fig_compare.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Relative Performance (Base=100)",
    legend_title="Asset",
    margin=dict(l=0, r=0, t=50, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_compare, use_container_width=True)

# Calculate relative performance
ticker_return = combined_data[ticker_symbol].iloc[-1] - 100
spy_return = combined_data['S&P 500'].iloc[-1] - 100
qqq_return = combined_data['NASDAQ'].iloc[-1] - 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"{ticker_symbol} Return", f"{ticker_return:.2f}%")
with col2:
    st.metric("S&P 500 Return", f"{spy_return:.2f}%", f"{ticker_return - spy_return:.2f}%")
with col3:
    st.metric("NASDAQ Return", f"{qqq_return:.2f}%", f"{ticker_return - qqq_return:.2f}%")

# Trading strategy section
st.markdown("## Simple Trading Strategy Analysis")
st.markdown("Basic moving average crossover strategy simulation")

# Calculate moving averages
stock_data_strategy = stock_data.copy()
stock_data_strategy['MA20'] = stock_data_strategy['Close'].rolling(window=20).mean()
stock_data_strategy['MA50'] = stock_data_strategy['Close'].rolling(window=50).mean()

# Generate trading signals
stock_data_strategy['Signal'] = 0
stock_data_strategy.loc[stock_data_strategy['MA20'] > stock_data_strategy['MA50'], 'Signal'] = 1
stock_data_strategy.loc[stock_data_strategy['MA20'] < stock_data_strategy['MA50'], 'Signal'] = -1

# Create strategy returns
stock_data_strategy['Return'] = stock_data_strategy['Close'].pct_change()
stock_data_strategy['Strategy Return'] = stock_data_strategy['Signal'].shift(1) * stock_data_strategy['Return']

# Calculate cumulative returns
stock_data_strategy['Cumulative Return'] = (1 + stock_data_strategy['Return']).cumprod()
stock_data_strategy['Cumulative Strategy Return'] = (1 + stock_data_strategy['Strategy Return']).cumprod()

# Only keep valid data
stock_data_strategy = stock_data_strategy.dropna()

# Display strategy chart
if len(stock_data_strategy) > 0:
    strategy_data = stock_data_strategy.reset_index()

    fig_strategy = go.Figure()

    # Add buy and hold line
    fig_strategy.add_trace(
        go.Scatter(
            x=strategy_data['index'],
            y=strategy_data['Cumulative Return'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#6c757d', width=2)
        )
    )

    # Add strategy line
    fig_strategy.add_trace(
        go.Scatter(
            x=strategy_data['index'],
            y=strategy_data['Cumulative Strategy Return'],
            mode='lines',
            name='MA Crossover Strategy',
            line=dict(color='#2962ff', width=2)
        )
    )

    # Improve chart formatting
    fig_strategy.update_layout(
        title="Strategy Performance Comparison",
        height=400,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (Base=1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white"
    )

    st.plotly_chart(fig_strategy, use_container_width=True)

    # Strategy statistics
    if len(stock_data_strategy) > 0:
        col1, col2, col3, col4 = st.columns(4)

        # Buy & hold return
        buy_hold_return = stock_data_strategy['Cumulative Return'].iloc[-1] - 1
        with col1:
            st.metric("Buy & Hold Return", f"{buy_hold_return * 100:.2f}%")

        # Strategy return
        strategy_return = stock_data_strategy['Cumulative Strategy Return'].iloc[-1] - 1
        with col2:
            st.metric("Strategy Return",
                      f"{strategy_return * 100:.2f}%",
                      f"{(strategy_return - buy_hold_return) * 100:.2f}%")

        # Sharpe ratio calculation
        strategy_daily_returns = stock_data_strategy['Strategy Return'].dropna()
        strategy_sharpe = strategy_daily_returns.mean() / strategy_daily_returns.std() * np.sqrt(252)

        with col3:
            st.metric("Strategy Sharpe", f"{strategy_sharpe:.2f}")

        # Maximum drawdown calculation for strategy
        cum_returns = stock_data_strategy['Cumulative Strategy Return']
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1) * 100
        max_dd = drawdown.min()

        with col4:
            st.metric("Strategy Max Drawdown", f"{max_dd:.2f}%")


def format_metric(value):
    """Format metric values with appropriate precision and symbols"""
    if value is None:
        return "N/A"
    elif isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.4f}"
        elif abs(value) < 1:
            return f"{value:.2f}"
        elif abs(value) < 1000000:
            return f"{value:,.2f}"
        else:
            return f"{value / 1000000:,.2f}M"
    else:
        return str(value)
