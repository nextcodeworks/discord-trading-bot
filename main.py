import os
import discord
from discord.ext import commands
from discord import Embed
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import json
import hashlib
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from io import BytesIO
import atexit

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Constants
BOT_PREFIX = '!'
EMBED_COLOR = 0x000000  # Black theme
WHITE_COLOR = 0xFFFFFF
BOT_IMAGE_URL = "https://avatars.githubusercontent.com/u/195336508?v=4"
NEXT_CODE_WORKS_URL = "https://nextcodeworks.github.io/"
GITHUB_PROFILE_URL = "https://github.com/nextcodeworks"

# Initialize bot
intents = discord.Intents.default()
intents.message_content = True  # This requires the Message Content Intent to be enabled
bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)

class TradingBot:
    """Main trading bot class that handles all trading functionality"""
    
    def __init__(self):
        self.paper_trading_portfolios = {}  # user_id: portfolio_dict
        self.user_strategies = {}  # user_id: strategy_dict
        self.market_data_cache = {}
        self.session = None  # Will be initialized later
        
    async def init_session(self):
        """Initialize the aiohttp session"""
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        if self.session:
            await self.session.close()

# Initialize trading bot
bot.trading_bot = TradingBot()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print('------')
    
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(embed=create_embed("Error", "Command not found. Use !help for available commands.", color=0xFF0000))
    else:
        await ctx.send(embed=create_embed("Error", f"An error occurred: {str(error)}", color=0xFF0000))

def create_embed(title: str, description: str, color: int = EMBED_COLOR, fields: Optional[List[Tuple[str, str, bool]]] = None) -> Embed:
    """Create a styled embed message with black-and-white theme"""
    embed = Embed(title=title, description=description, color=color)
    if fields:
        for name, value, inline in fields:
            embed.add_field(name=name, value=value, inline=inline)
    embed.set_footer(text="Trading Bot | Next Code Works", icon_url=BOT_IMAGE_URL)
    return embed

async def fetch_market_data(session: aiohttp.ClientSession, symbol: str, exchange: str = "binance") -> Dict:
    """Fetch market data from specified exchange"""
    try:
        if exchange.lower() == "binance":
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            async with session.get(url) as response:
                data = await response.json()
                return {
                    'symbol': data['symbol'],
                    'price': float(data['lastPrice']),
                    'volume': float(data['volume']),
                    'price_change': float(data['priceChange']),
                    'price_change_percent': float(data['priceChangePercent']),
                    'high': float(data['highPrice']),
                    'low': float(data['lowPrice']),
                    'exchange': 'Binance'
                }
        # Add more exchanges here
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return None
    
class ExchangeAPI:
    """Handles all exchange API interactions"""
    
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.exchanges = {
            'binance': {
                'base_url': 'https://api.binance.com/api/v3',
                'symbols_endpoint': '/exchangeInfo',
                'ticker_endpoint': '/ticker/24hr',
                'klines_endpoint': '/klines'
            },
            # Can add more exchanges here
        }
    
    async def get_market_data(self, symbol: str, exchange: str = 'binance') -> Dict:
        """Get detailed market data for a symbol"""
        try:
            if exchange.lower() == "binance":
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                async with self.session.get(url) as response:
                    data = await response.json()
                    return {
                        'symbol': data['symbol'],
                        'price': float(data['lastPrice']),
                        'volume': float(data['volume']),
                        'price_change': float(data['priceChange']),
                        'price_change_percent': float(data['priceChangePercent']),
                        'high': float(data['highPrice']),
                        'low': float(data['lowPrice']),
                        'exchange': 'Binance'
                    }
            # Add more exchanges here if needed
            return None
        except Exception as e:
            print(f"Error getting market data: {e}")
            return None
    
    async def get_available_symbols(self, exchange: str) -> List[str]:
        """Get available trading symbols/pairs from exchange"""
        try:
            exchange_config = self.exchanges.get(exchange.lower())
            if not exchange_config:
                return []
                
            url = exchange_config['base_url'] + exchange_config['symbols_endpoint']
            async with self.session.get(url) as response:
                data = await response.json()
                return [symbol['symbol'] for symbol in data['symbols']]
        except Exception as e:
            print(f"Error getting symbols from {exchange}: {e}")
            return []
    
    async def get_historical_data(self, symbol: str, interval: str = '1d', limit: int = 100, exchange: str = 'binance') -> pd.DataFrame:
        """Get historical candlestick data"""
        try:
            exchange_config = self.exchanges.get(exchange.lower())
            if not exchange_config:
                return pd.DataFrame()
                
            url = f"{exchange_config['base_url']}{exchange_config['klines_endpoint']}?symbol={symbol}&interval={interval}&limit={limit}"
            async with self.session.get(url) as response:
                data = await response.json()
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                # Convert columns to numeric
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                return df
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    async def get_current_price(self, symbol: str, exchange: str = 'binance') -> Optional[float]:
        """Get current price for a symbol"""
        try:
            exchange_config = self.exchanges.get(exchange.lower())
            if not exchange_config:
                return None
                
            url = f"{exchange_config['base_url']}{exchange_config['ticker_endpoint']}?symbol={symbol}"
            async with self.session.get(url) as response:
                data = await response.json()
                return float(data['lastPrice'])
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None
        
@bot.command(name='price', help='Get current price for a cryptocurrency pair')
async def get_price(ctx, symbol: str, exchange: str = 'binance'):
    """Get current price for a cryptocurrency pair"""
    await ctx.typing()
    
    # Validate symbol format (basic validation)
    if not symbol.isalnum():
        await ctx.send(embed=create_embed("Error", "Invalid symbol format. Symbols should be alphanumeric (e.g., BTCUSDT).", color=0xFF0000))
        return
    
    price = await bot.trading_bot.exchange_api.get_current_price(symbol, exchange)
    if price is None:
        await ctx.send(embed=create_embed("Error", f"Could not fetch price for {symbol} on {exchange}. Please check the symbol and try again.", color=0xFF0000))
    else:
        embed = create_embed(
            title=f"{symbol} Price on {exchange.capitalize()}",
            description=f"Current price: **{price:.8f}**",
            fields=[
                ("Symbol", symbol, True),
                ("Exchange", exchange.capitalize(), True),
                ("Last Updated", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), True)
            ]
        )
        await ctx.send(embed=embed)

@bot.command(name='market', help='Get detailed market data for a cryptocurrency pair')
async def get_market_data(ctx, symbol: str, exchange: str = 'binance'):
    """Get detailed market data for a cryptocurrency pair"""
    await ctx.typing()
    
    data = await bot.trading_bot.exchange_api.get_market_data(symbol, exchange)
    if not data:
        await ctx.send(embed=create_embed("Error", f"Could not fetch market data for {symbol} on {exchange}. Please check the symbol and try again.", color=0xFF0000))
        return
    
    # Create detailed embed
    fields = [
        ("Symbol", data['symbol'], True),
        ("Exchange", data['exchange'], True),
        ("Price", f"{data['price']:.8f}", True),
        ("24h Change", f"{data['price_change']:.4f} ({data['price_change_percent']:.2f}%)", True),
        ("24h High", f"{data['high']:.8f}", True),
        ("24h Low", f"{data['low']:.8f}", True),
        ("24h Volume", f"{data['volume']:.2f}", True),
        ("Last Updated", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), False)
    ]
    
    embed = create_embed(
        title=f"Market Data for {symbol} on {exchange.capitalize()}",
        description="Detailed market statistics",
        fields=fields
    )
    
    await ctx.send(embed=embed)

@bot.command(name='symbols', help='List available trading symbols on an exchange')
async def list_symbols(ctx, exchange: str = 'binance', page: int = 1):
    """List available trading symbols on an exchange"""
    await ctx.typing()
    
    symbols = await bot.trading_bot.exchange_api.get_available_symbols(exchange)
    if not symbols:
        await ctx.send(embed=create_embed("Error", f"Could not fetch symbols from {exchange}. The exchange may not be supported.", color=0xFF0000))
        return
    
    # Paginate results
    items_per_page = 20
    total_pages = (len(symbols) + items_per_page - 1) // items_per_page
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    symbol_list = "\n".join(symbols[start_idx:end_idx])
    
    embed = create_embed(
        title=f"Available Symbols on {exchange.capitalize()}",
        description=f"Page {page} of {total_pages}",
        fields=[("Symbols", symbol_list, False)]
    )
    
    await ctx.send(embed=embed)
    
class TradingSignals:
    """Generates trading signals based on technical indicators"""
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> Dict:
        """Generate trading signals from historical data"""
        signals = {
            'timestamp': datetime.utcnow().isoformat(),
            'indicators': {}
        }
        
        try:
            # Calculate indicators
            close_prices = df['close'].values.astype(float)  # Ensure float type
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            volume = df['volume'].values.astype(float)
            
            # RSI
            rsi_period = 14
            rsi = talib.RSI(close_prices, timeperiod=rsi_period)
            signals['indicators']['RSI'] = {
                'value': float(rsi[-1]) if not np.isnan(rsi[-1]) else 0.0,
                'signal': 'neutral'
            }
            if not np.isnan(rsi[-1]):
                if rsi[-1] > 70:
                    signals['indicators']['RSI']['signal'] = 'overbought'
                elif rsi[-1] < 30:
                    signals['indicators']['RSI']['signal'] = 'oversold'
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            signals['indicators']['MACD'] = {
                'macd': float(macd[-1]) if not np.isnan(macd[-1]) else 0.0,
                'signal': float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0.0,
                'hist': float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0,
                'signal_cross': 'neutral'
            }
            if not np.isnan(macd[-1]) and not np.isnan(macd_signal[-1]):
                if macd[-1] > macd_signal[-1]:
                    signals['indicators']['MACD']['signal_cross'] = 'bullish'
                else:
                    signals['indicators']['MACD']['signal_cross'] = 'bearish'
            
            # Moving Averages
            sma_50 = talib.SMA(close_prices, timeperiod=50)
            sma_200 = talib.SMA(close_prices, timeperiod=200)
            signals['indicators']['MA'] = {
                'SMA_50': float(sma_50[-1]) if not np.isnan(sma_50[-1]) else 0.0,
                'SMA_200': float(sma_200[-1]) if not np.isnan(sma_200[-1]) else 0.0,
                'signal': 'neutral'
            }
            if not np.isnan(sma_50[-1]) and not np.isnan(sma_200[-1]):
                if sma_50[-1] > sma_200[-1]:
                    signals['indicators']['MA']['signal'] = 'golden_cross'
                elif sma_50[-1] < sma_200[-1]:
                    signals['indicators']['MA']['signal'] = 'death_cross'
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
            signals['indicators']['Bollinger_Bands'] = {
                'upper': float(upper[-1]) if not np.isnan(upper[-1]) else 0.0,
                'middle': float(middle[-1]) if not np.isnan(middle[-1]) else 0.0,
                'lower': float(lower[-1]) if not np.isnan(lower[-1]) else 0.0,
                'signal': 'neutral'
            }
            if not np.isnan(close_prices[-1]):
                if close_prices[-1] > upper[-1]:
                    signals['indicators']['Bollinger_Bands']['signal'] = 'overbought'
                elif close_prices[-1] < lower[-1]:
                    signals['indicators']['Bollinger_Bands']['signal'] = 'oversold'
            
            # Generate overall signal
            buy_signals = 0
            sell_signals = 0
            
            for indicator in signals['indicators'].values():
                if indicator.get('signal') in ['bullish', 'oversold', 'golden_cross']:
                    buy_signals += 1
                elif indicator.get('signal') in ['bearish', 'overbought', 'death_cross']:
                    sell_signals += 1
            
            if buy_signals > sell_signals:
                signals['overall_signal'] = 'BUY'
            elif sell_signals > buy_signals:
                signals['overall_signal'] = 'SELL'
            else:
                signals['overall_signal'] = 'NEUTRAL'
            
            return signals
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return None

@bot.command(name='signals', help='Generate trading signals for a cryptocurrency pair')
async def generate_signals(ctx, symbol: str, interval: str = '1d', limit: int = 100, exchange: str = 'binance'):
    """Generate trading signals for a cryptocurrency pair"""
    await ctx.typing()
    
    # Validate interval
    valid_intervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if interval not in valid_intervals:
        await ctx.send(embed=create_embed(
            "Error",
            f"Invalid interval. Valid intervals are: {', '.join(valid_intervals)}",
            color=0xFF0000
        ))
        return
    
    # Get historical data
    df = await bot.trading_bot.exchange_api.get_historical_data(symbol, interval, limit, exchange)
    if df.empty:
        await ctx.send(embed=create_embed("Error", f"Could not fetch historical data for {symbol} on {exchange}.", color=0xFF0000))
        return
    
    # Generate signals
    signals = TradingSignals.generate_signals(df)
    if not signals:
        await ctx.send(embed=create_embed("Error", "Could not generate trading signals.", color=0xFF0000))
        return
    
    # Prepare signal fields
    fields = []
    
    # Overall signal
    signal_emoji = "⚪"  # Neutral
    if signals['overall_signal'] == 'BUY':
        signal_emoji = "🟢"
    elif signals['overall_signal'] == 'SELL':
        signal_emoji = "🔴"
    
    fields.append(("Overall Signal", f"{signal_emoji} {signals['overall_signal']}", False))
    
    # Individual indicators
    for indicator, data in signals['indicators'].items():
        value_str = ""
        signal_emoji = "⚪"  # Neutral
        
        if indicator == 'RSI':
            value_str = f"{data['value']:.2f}"
            if data['signal'] == 'overbought':
                signal_emoji = "🔴"
            elif data['signal'] == 'oversold':
                signal_emoji = "🟢"
        elif indicator == 'MACD':
            value_str = f"MACD: {data['macd']:.4f}\nSignal: {data['signal']:.4f}\nHist: {data['hist']:.4f}"
            if data['signal'] == 'bullish':
                signal_emoji = "🟢"
            elif data['signal'] == 'bearish':
                signal_emoji = "🔴"
        elif indicator == 'MA':
            value_str = f"SMA 50: {data['SMA_50']:.4f}\nSMA 200: {data['SMA_200']:.4f}"
            if data['signal'] == 'golden_cross':
                signal_emoji = "🟢"
            elif data['signal'] == 'death_cross':
                signal_emoji = "🔴"
        elif indicator == 'Bollinger_Bands':
            value_str = f"Upper: {data['upper']:.4f}\nMiddle: {data['middle']:.4f}\nLower: {data['lower']:.4f}"
            if data['signal'] == 'overbought':
                signal_emoji = "🔴"
            elif data['signal'] == 'oversold':
                signal_emoji = "🟢"
        
        fields.append((f"{indicator} {signal_emoji}", value_str, True))
    
    # Add additional info
    fields.extend([
        ("Symbol", symbol, True),
        ("Exchange", exchange.capitalize(), True),
        ("Interval", interval, True),
        ("Data Points", str(limit), True),
        ("Analysis Date", signals['timestamp'], False)
    ])
    
    embed = create_embed(
        title=f"Trading Signals for {symbol}",
        description="Technical analysis based on multiple indicators",
        fields=fields
    )
    
    await ctx.send(embed=embed)
    
    
class PortfolioManager:
    """Manages user portfolios including paper trading"""
    
    def __init__(self):
        self.portfolios = {}  # user_id: portfolio
        self.paper_trading_portfolios = {}  # user_id: paper_portfolio
        self.transaction_history = {}  # user_id: [transactions]
        
    def init_user_portfolio(self, user_id: str):
        """Initialize a user's portfolio if it doesn't exist"""
        if user_id not in self.portfolios:
            self.portfolios[user_id] = {
                'balances': {},
                'total_value': 0.0,
                'last_updated': datetime.utcnow().isoformat()
            }
        if user_id not in self.paper_trading_portfolios:
            self.paper_trading_portfolios[user_id] = {
                'balances': {'USDT': 10000.0},  # Starting with 10,000 USDT for paper trading
                'total_value': 10000.0,
                'open_orders': [],
                'last_updated': datetime.utcnow().isoformat()
            }
        if user_id not in self.transaction_history:
            self.transaction_history[user_id] = []
    
    def get_portfolio(self, user_id: str, paper: bool = False) -> Dict:
        """Get user's portfolio"""
        self.init_user_portfolio(user_id)
        return self.paper_trading_portfolios[user_id] if paper else self.portfolios[user_id]
    
    def add_transaction(self, user_id: str, transaction: Dict):
        """Add a transaction to user's history"""
        self.init_user_portfolio(user_id)
        self.transaction_history[user_id].append(transaction)
    
    def get_transaction_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's transaction history"""
        self.init_user_portfolio(user_id)
        return self.transaction_history[user_id][-limit:]
    
    async def update_portfolio_value(self, user_id: str, session: aiohttp.ClientSession, paper: bool = False):
        """Update portfolio total value based on current prices"""
        self.init_user_portfolio(user_id)
        portfolio = self.paper_trading_portfolios[user_id] if paper else self.portfolios[user_id]
        
        total_value = 0.0
        for asset, amount in portfolio['balances'].items():
            if asset == 'USDT':
                total_value += amount
            else:
                # For other assets, we need to get their value in USDT
                symbol = f"{asset}USDT"
                price = await ExchangeAPI(session).get_current_price(symbol)
                if price:
                    total_value += amount * price
        
        portfolio['total_value'] = total_value
        portfolio['last_updated'] = datetime.utcnow().isoformat()
        return portfolio

@bot.command(name='portfolio', help='View your trading portfolio')
async def view_portfolio(ctx, paper: Optional[str] = None):
    """View your trading portfolio (real or paper trading)"""
    await ctx.typing()
    
    is_paper = paper and paper.lower() in ['paper', 'p']
    portfolio_type = "Paper Trading" if is_paper else "Real Trading"
    
    # Update portfolio value with current prices
    portfolio = await bot.trading_bot.portfolio_manager.update_portfolio_value(str(ctx.author.id), bot.trading_bot.session, is_paper)
    
    # Prepare portfolio display
    if not portfolio['balances']:
        await ctx.send(embed=create_embed(
            f"{portfolio_type} Portfolio",
            "Your portfolio is empty.",
            fields=[("Total Value", "0.00 USDT", False)]
        ))
        return
    
    # Sort balances by value
    sorted_balances = sorted(
        portfolio['balances'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Create balance fields
    fields = []
    for asset, amount in sorted_balances:
        if asset == 'USDT':
            fields.append((asset, f"{amount:.2f}", True))
        else:
            # Get current price for non-USDT assets
            price = await bot.trading_bot.exchange_api.get_current_price(f"{asset}USDT")
            if price:
                value = amount * price
                fields.append((asset, f"{amount:.8f}\n≈ {value:.2f} USDT", True))
            else:
                fields.append((asset, f"{amount:.8f}\n(Price unavailable)", True))
    
    # Add total value
    fields.append(("Total Portfolio Value", f"{portfolio['total_value']:.2f} USDT", False))
    fields.append(("Last Updated", portfolio['last_updated'], False))
    
    embed = create_embed(
        title=f"{portfolio_type} Portfolio",
        description=f"Portfolio breakdown for {ctx.author.display_name}",
        fields=fields
    )
    
    await ctx.send(embed=embed)

@bot.command(name='transactions', help='View your transaction history')
async def view_transactions(ctx, limit: int = 10, paper: Optional[str] = None):
    """View your transaction history"""
    await ctx.typing()
    
    is_paper = paper and paper.lower() in ['paper', 'p']
    portfolio_type = "Paper Trading" if is_paper else "Real Trading"
    
    transactions = bot.trading_bot.portfolio_manager.get_transaction_history(str(ctx.author.id), limit)
    
    if not transactions:
        await ctx.send(embed=create_embed(
            f"{portfolio_type} Transactions",
            "You have no transactions yet.",
            color=WHITE_COLOR
        ))
        return
    
    # Format transactions
    transaction_list = []
    for tx in reversed(transactions):  # Show newest first
        tx_type = tx.get('type', 'unknown').upper()
        symbol = tx.get('symbol', 'N/A')
        amount = tx.get('amount', 0)
        price = tx.get('price', 0)
        total = amount * price
        timestamp = tx.get('timestamp', 'N/A')
        
        transaction_list.append(
            f"**{tx_type}** {symbol} | {amount:.8f} @ {price:.8f} = {total:.2f} USDT\n"
            f"*{timestamp}*"
        )
    
    embed = create_embed(
        title=f"{portfolio_type} Transactions (Last {limit})",
        description="\n\n".join(transaction_list),
        color=WHITE_COLOR
    )
    
    await ctx.send(embed=embed)
    
@bot.command(name='buy', help='Execute a buy order')
async def buy_order(ctx, symbol: str, amount: float, price: Optional[float] = None, paper: Optional[str] = None):
    """Execute a buy order (real or paper trading)"""
    await ctx.typing()
    
    is_paper = paper and paper.lower() in ['paper', 'p']
    
    if amount <= 0:
        await ctx.send(embed=create_embed("Error", "Amount must be positive.", color=0xFF0000))
        return
    
    # For paper trading, we'll simulate the order
    if is_paper:
        portfolio = bot.trading_bot.portfolio_manager.get_portfolio(str(ctx.author.id), True)
        
        # Get current price if not specified
        if price is None:
            current_price = await bot.trading_bot.exchange_api.get_current_price(symbol)
            if current_price is None:
                await ctx.send(embed=create_embed("Error", f"Could not fetch price for {symbol}. Please specify a price.", color=0xFF0000))
                return
            price = current_price
        
        total_cost = amount * price
        
        # Check if user has enough USDT
        if portfolio['balances'].get('USDT', 0) < total_cost:
            await ctx.send(embed=create_embed("Error", "Insufficient USDT balance for this order.", color=0xFF0000))
            return
        
        # Execute paper trade
        base_asset = symbol.replace('USDT', '')
        
        # Deduct USDT
        portfolio['balances']['USDT'] -= total_cost
        
        # Add bought asset
        if base_asset in portfolio['balances']:
            portfolio['balances'][base_asset] += amount
        else:
            portfolio['balances'][base_asset] = amount
        
        # Record transaction
        transaction = {
            'type': 'buy',
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'total': total_cost,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'filled'
        }
        bot.trading_bot.portfolio_manager.add_transaction(str(ctx.author.id), transaction)
        
        # Update portfolio value
        await bot.trading_bot.portfolio_manager.update_portfolio_value(str(ctx.author.id), bot.trading_bot.session, True)
        
        # Send confirmation
        embed = create_embed(
            title="Paper Trading Order Filled",
            description=f"Successfully executed paper trade for {symbol}",
            fields=[
                ("Type", "BUY", True),
                ("Symbol", symbol, True),
                ("Amount", f"{amount:.8f}", True),
                ("Price", f"{price:.8f} USDT", True),
                ("Total", f"{total_cost:.2f} USDT", True),
                ("Status", "Filled", True),
                ("Transaction Date", transaction['timestamp'], False)
            ]
        )
        await ctx.send(embed=embed)
    else:
        # Real trading would go here (would require API keys)
        await ctx.send(embed=create_embed(
            "Error",
            "Real trading is not implemented in this version. Use paper trading with '!buy <symbol> <amount> paper'.",
            color=0xFF0000
        ))

@bot.command(name='sell', help='Execute a sell order')
async def sell_order(ctx, symbol: str, amount: float, price: Optional[float] = None, paper: Optional[str] = None):
    """Execute a sell order (real or paper trading)"""
    await ctx.typing()
    
    is_paper = paper and paper.lower() in ['paper', 'p']
    
    if amount <= 0:
        await ctx.send(embed=create_embed("Error", "Amount must be positive.", color=0xFF0000))
        return
    
    # For paper trading, we'll simulate the order
    if is_paper:
        portfolio = bot.trading_bot.portfolio_manager.get_portfolio(str(ctx.author.id), True)
        base_asset = symbol.replace('USDT', '')
        
        # Check if user has enough of the asset
        if portfolio['balances'].get(base_asset, 0) < amount:
            await ctx.send(embed=create_embed("Error", f"Insufficient {base_asset} balance for this order.", color=0xFF0000))
            return
        
        # Get current price if not specified
        if price is None:
            current_price = await bot.trading_bot.exchange_api.get_current_price(symbol)
            if current_price is None:
                await ctx.send(embed=create_embed("Error", f"Could not fetch price for {symbol}. Please specify a price.", color=0xFF0000))
                return
            price = current_price
        
        total_value = amount * price
        
        # Execute paper trade
        # Deduct sold asset
        portfolio['balances'][base_asset] -= amount
        if portfolio['balances'][base_asset] <= 0.00000001:  # Clean up tiny balances
            del portfolio['balances'][base_asset]
        
        # Add USDT
        portfolio['balances']['USDT'] = portfolio['balances'].get('USDT', 0) + total_value
        
        # Record transaction
        transaction = {
            'type': 'sell',
            'symbol': symbol,
            'amount': amount,
            'price': price,
            'total': total_value,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'filled'
        }
        bot.trading_bot.portfolio_manager.add_transaction(str(ctx.author.id), transaction)
        
        # Update portfolio value
        await bot.trading_bot.portfolio_manager.update_portfolio_value(str(ctx.author.id), bot.trading_bot.session, True)
        
        # Send confirmation
        embed = create_embed(
            title="Paper Trading Order Filled",
            description=f"Successfully executed paper trade for {symbol}",
            fields=[
                ("Type", "SELL", True),
                ("Symbol", symbol, True),
                ("Amount", f"{amount:.8f}", True),
                ("Price", f"{price:.8f} USDT", True),
                ("Total", f"{total_value:.2f} USDT", True),
                ("Status", "Filled", True),
                ("Transaction Date", transaction['timestamp'], False)
            ]
        )
        await ctx.send(embed=embed)
    else:
        # Real trading would go here (would require API keys)
        await ctx.send(embed=create_embed(
            "Error",
            "Real trading is not implemented in this version. Use paper trading with '!sell <symbol> <amount> paper'.",
            color=0xFF0000
        ))
        
class Backtester:
    """Handles backtesting of trading strategies"""
    
    @staticmethod
    async def backtest_strategy(session: aiohttp.ClientSession, symbol: str, strategy: str, 
                              interval: str = '1d', limit: int = 100, exchange: str = 'binance',
                              initial_balance: float = 10000.0) -> Dict:
        """Backtest a trading strategy on historical data"""
        try:
            # Get historical data
            df = await ExchangeAPI(session).get_historical_data(symbol, interval, limit, exchange)
            if df.empty:
                return None
            
            # Prepare results
            results = {
                'symbol': symbol,
                'exchange': exchange,
                'interval': interval,
                'strategy': strategy,
                'initial_balance': initial_balance,
                'final_balance': initial_balance,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'start_date': df['open_time'].iloc[0].strftime('%Y-%m-%d'),
                'end_date': df['open_time'].iloc[-1].strftime('%Y-%m-%d'),
                'trades': []
            }
            
            # Simple moving average crossover strategy
            if strategy.lower() == 'sma_crossover':
                df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
                df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
                
                position = None
                entry_price = 0.0
                balance = initial_balance
                max_balance = initial_balance
                
                for i in range(200, len(df)):
                    current_price = df['close'].iloc[i]
                    sma_50 = df['SMA_50'].iloc[i]
                    sma_200 = df['SMA_200'].iloc[i]
                    
                    # Buy signal (50 crosses above 200)
                    if sma_50 > sma_200 and (i > 0 and df['SMA_50'].iloc[i-1] <= df['SMA_200'].iloc[i-1]):
                        if position == 'short':
                            # Close short position
                            pnl = (entry_price - current_price) / entry_price
                            balance *= (1 + pnl)
                            results['trades'].append({
                                'type': 'close_short',
                                'date': df['open_time'].iloc[i].strftime('%Y-%m-%d'),
                                'price': current_price,
                                'pnl': pnl,
                                'balance': balance
                            })
                            if pnl > 0:
                                results['winning_trades'] += 1
                            else:
                                results['losing_trades'] += 1
                        
                        # Open long position
                        position = 'long'
                        entry_price = current_price
                        results['trades'].append({
                            'type': 'open_long',
                            'date': df['open_time'].iloc[i].strftime('%Y-%m-%d'),
                            'price': current_price,
                            'balance': balance
                        })
                        results['total_trades'] += 1
                    
                    # Sell signal (50 crosses below 200)
                    elif sma_50 < sma_200 and (i > 0 and df['SMA_50'].iloc[i-1] >= df['SMA_200'].iloc[i-1]):
                        if position == 'long':
                            # Close long position
                            pnl = (current_price - entry_price) / entry_price
                            balance *= (1 + pnl)
                            results['trades'].append({
                                'type': 'close_long',
                                'date': df['open_time'].iloc[i].strftime('%Y-%m-%d'),
                                'price': current_price,
                                'pnl': pnl,
                                'balance': balance
                            })
                            if pnl > 0:
                                results['winning_trades'] += 1
                            else:
                                results['losing_trades'] += 1
                        
                        # Open short position
                        position = 'short'
                        entry_price = current_price
                        results['trades'].append({
                            'type': 'open_short',
                            'date': df['open_time'].iloc[i].strftime('%Y-%m-%d'),
                            'price': current_price,
                            'balance': balance
                        })
                        results['total_trades'] += 1
                    
                    # Update max drawdown
                    max_balance = max(max_balance, balance)
                    current_drawdown = (max_balance - balance) / max_balance
                    results['max_drawdown'] = max(results['max_drawdown'], current_drawdown)
                
                # Close any open position at the end
                if position == 'long':
                    current_price = df['close'].iloc[-1]
                    pnl = (current_price - entry_price) / entry_price
                    balance *= (1 + pnl)
                    results['trades'].append({
                        'type': 'close_long',
                        'date': df['open_time'].iloc[-1].strftime('%Y-%m-%d'),
                        'price': current_price,
                        'pnl': pnl,
                        'balance': balance
                    })
                    if pnl > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
                elif position == 'short':
                    current_price = df['close'].iloc[-1]
                    pnl = (entry_price - current_price) / entry_price
                    balance *= (1 + pnl)
                    results['trades'].append({
                        'type': 'close_short',
                        'date': df['open_time'].iloc[-1].strftime('%Y-%m-%d'),
                        'price': current_price,
                        'pnl': pnl,
                        'balance': balance
                    })
                    if pnl > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
                
                results['final_balance'] = balance
                results['profit'] = (balance - initial_balance) / initial_balance * 100
                
                # Calculate Sharpe ratio (simplified)
                if results['total_trades'] > 0:
                    avg_return = results['profit'] / results['total_trades']
                    results['sharpe_ratio'] = avg_return / (results['max_drawdown'] + 0.0001)  # Avoid division by zero
            
            return results
            
        except Exception as e:
            print(f"Error in backtesting: {e}")
            return None

@bot.command(name='backtest', help='Backtest a trading strategy on historical data')
async def backtest_strategy(ctx, symbol: str, strategy: str = 'sma_crossover', 
                          interval: str = '1d', limit: int = 100, exchange: str = 'binance'):
    """Backtest a trading strategy on historical data"""
    await ctx.typing()
    
    # Validate strategy
    valid_strategies = ['sma_crossover']  # Can add more strategies
    if strategy.lower() not in valid_strategies:
        await ctx.send(embed=create_embed(
            "Error",
            f"Invalid strategy. Valid strategies are: {', '.join(valid_strategies)}",
            color=0xFF0000
        ))
        return
    
    # Run backtest
    results = await Backtester.backtest_strategy(
        bot.trading_bot.session, symbol, strategy, interval, limit, exchange
    )
    
    if not results:
        await ctx.send(embed=create_embed("Error", "Backtesting failed. Please check your parameters and try again.", color=0xFF0000))
        return
    
    # Prepare performance summary
    profit_color = WHITE_COLOR
    if results['profit'] > 0:
        profit_color = 0x00FF00  # Green
    elif results['profit'] < 0:
        profit_color = 0xFF0000  # Red
    
    fields = [
        ("Symbol", results['symbol'], True),
        ("Exchange", results['exchange'].capitalize(), True),
        ("Strategy", results['strategy'].replace('_', ' ').title(), True),
        ("Period", f"{results['start_date']} to {results['end_date']}", True),
        ("Initial Balance", f"{results['initial_balance']:.2f} USDT", True),
        ("Final Balance", f"{results['final_balance']:.2f} USDT", True),
        ("Profit/Loss", f"{results['profit']:.2f}%", True),
        ("Total Trades", str(results['total_trades']), True),
        ("Winning Trades", f"{results['winning_trades']} ({results['winning_trades']/results['total_trades']*100:.1f}%)" if results['total_trades'] > 0 else "0", True),
        ("Losing Trades", f"{results['losing_trades']} ({results['losing_trades']/results['total_trades']*100:.1f}%)" if results['total_trades'] > 0 else "0", True),
        ("Max Drawdown", f"{results['max_drawdown']*100:.2f}%", True),
        ("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}", True),
        ("Timeframe", interval, True),
        ("Data Points", str(limit), True)
    ]
    
    # Create equity curve plot
    plt.figure(figsize=(10, 5))
    balances = [trade['balance'] for trade in results['trades'] if 'balance' in trade]
    if balances:
        plt.plot(balances)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Balance (USDT)')
        plt.grid(True)
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Create embed with image
        embed = create_embed(
            title=f"Backtest Results for {symbol}",
            description=f"Performance of {results['strategy'].replace('_', ' ')} strategy",
            color=profit_color,
            fields=fields
        )
        file = discord.File(buf, filename='equity_curve.png')
        embed.set_image(url="attachment://equity_curve.png")
        await ctx.send(file=file, embed=embed)
    else:
        # No trades were made
        embed = create_embed(
            title=f"Backtest Results for {symbol}",
            description=f"No trades were executed using the {results['strategy'].replace('_', ' ')} strategy during this period.",
            color=profit_color,
            fields=fields
        )
        await ctx.send(embed=embed)
        
@bot.command(name='paper', help='Paper trading commands')
async def paper_trading(ctx, action: str = 'status', *args):
    """Paper trading commands"""
    await ctx.typing()
    
    action = action.lower()
    user_id = str(ctx.author.id)
    
    if action == 'status':
        # Show paper trading status
        portfolio = bot.trading_bot.portfolio_manager.get_portfolio(user_id, True)
        await bot.trading_bot.portfolio_manager.update_portfolio_value(user_id, bot.trading_bot.session, True)
        
        embed = create_embed(
            title="Paper Trading Status",
            description=f"Virtual portfolio for {ctx.author.display_name}",
            fields=[
                ("Total Value", f"{portfolio['total_value']:.2f} USDT", False),
                ("Available Balance", f"{portfolio['balances'].get('USDT', 0):.2f} USDT", True),
                ("Open Positions", str(len(portfolio['balances']) - 1 if 'USDT' in portfolio['balances'] else len(portfolio['balances'])), True),
                ("Last Updated", portfolio['last_updated'], False)
            ]
        )
        await ctx.send(embed=embed)
    
    elif action == 'reset':
        # Reset paper trading account
        bot.trading_bot.portfolio_manager.paper_trading_portfolios[user_id] = {
            'balances': {'USDT': 10000.0},
            'total_value': 10000.0,
            'open_orders': [],
            'last_updated': datetime.utcnow().isoformat()
        }
        bot.trading_bot.portfolio_manager.transaction_history[user_id] = []
        
        await ctx.send(embed=create_embed(
            "Paper Trading Reset",
            "Your paper trading account has been reset to 10,000 USDT.",
            color=WHITE_COLOR
        ))
    
    elif action == 'balance':
        # Show detailed balance
        portfolio = bot.trading_bot.portfolio_manager.get_portfolio(user_id, True)
        await bot.trading_bot.portfolio_manager.update_portfolio_value(user_id, bot.trading_bot.session, True)
        
        if not portfolio['balances']:
            await ctx.send(embed=create_embed(
                "Paper Trading Balance",
                "Your paper trading account is empty.",
                color=WHITE_COLOR
            ))
            return
        
        fields = []
        for asset, amount in portfolio['balances'].items():
            if asset == 'USDT':
                fields.append((asset, f"{amount:.2f}", True))
            else:
                price = await bot.trading_bot.exchange_api.get_current_price(f"{asset}USDT")
                if price:
                    value = amount * price
                    fields.append((asset, f"{amount:.8f}\n≈ {value:.2f} USDT", True))
                else:
                    fields.append((asset, f"{amount:.8f}\n(Price unavailable)", True))
        
        fields.append(("Total Value", f"{portfolio['total_value']:.2f} USDT", False))
        
        embed = create_embed(
            title="Paper Trading Balance",
            description="Detailed breakdown of your paper trading account",
            fields=fields
        )
        await ctx.send(embed=embed)
    
    else:
        await ctx.send(embed=create_embed(
            "Error",
            "Invalid paper trading command. Valid commands: status, reset, balance",
            color=0xFF0000
        ))
        
@bot.command(name='bothelp', help='Show all available commands')
async def show_help(ctx):
    """Show all available commands"""
    embed = create_embed(
        title="Trading Bot Help",
        description="Here are all the available commands:",
        fields=[
            ("📊 Market Data", "```!price <symbol> [exchange]\n!market <symbol> [exchange]\n!symbols [exchange] [page]```", False),
            ("📈 Trading Signals", "```!signals <symbol> [interval] [limit] [exchange]```", False),
            ("💰 Portfolio", "```!portfolio [paper]\n!transactions [limit] [paper]```", False),
            ("🛒 Trading", "```!buy <symbol> <amount> [price] [paper]\n!sell <symbol> <amount> [price] [paper]```", False),
            ("📝 Paper Trading", "```!paper [status|reset|balance]```", False),
            ("🧪 Backtesting", "```!backtest <symbol> [strategy] [interval] [limit] [exchange]```", False),
            ("ℹ️ About", "```!about```", False)
        ]
    )
    await ctx.send(embed=embed)

@bot.command(name='about', help='Show information about the bot')
async def about(ctx):
    """Show information about the bot"""
    embed = create_embed(
        title="About Trading Bot",
        description="A sophisticated cryptocurrency trading bot for Discord with real-time market data, trading signals, portfolio management, and paper trading capabilities.",
        color=WHITE_COLOR
    )
    embed.add_field(
        name="Features",
        value="• Real-time market data from multiple exchanges\n"
              "• Technical analysis and trading signals\n"
              "• Portfolio tracking and management\n"
              "• Paper trading simulation\n"
              "• Strategy backtesting\n"
              "• Clean, professional black-and-white interface",
        inline=False
    )
    embed.add_field(
        name="Links",
        value=f"[Next Code Works]({NEXT_CODE_WORKS_URL})\n"
              f"[GitHub Profile]({GITHUB_PROFILE_URL})",
        inline=False
    )
    embed.set_thumbnail(url=BOT_IMAGE_URL)
    await ctx.send(embed=embed)
    
class TradingBotError(Exception):
    """Base class for trading bot exceptions"""
    pass

class ExchangeAPIError(TradingBotError):
    """Errors related to exchange API calls"""
    pass

class InsufficientFundsError(TradingBotError):
    """Errors related to insufficient funds"""
    pass

class InvalidSymbolError(TradingBotError):
    """Errors related to invalid trading symbols"""
    pass

async def handle_trading_error(ctx, error: TradingBotError):
    """Handle trading-related errors and send appropriate Discord messages"""
    if isinstance(error, ExchangeAPIError):
        await ctx.send(embed=create_embed(
            "Exchange API Error",
            str(error),
            color=0xFF0000
        ))
    elif isinstance(error, InsufficientFundsError):
        await ctx.send(embed=create_embed(
            "Insufficient Funds",
            str(error),
            color=0xFF0000
        ))
    elif isinstance(error, InvalidSymbolError):
        await ctx.send(embed=create_embed(
            "Invalid Symbol",
            str(error),
            color=0xFF0000
        ))
    else:
        await ctx.send(embed=create_embed(
            "Trading Error",
            f"An unexpected error occurred: {str(error)}",
            color=0xFF0000
        ))
        
def encrypt_data(data: str, key: str) -> str:
    """Simple encryption for sensitive data"""
    # In a production environment, use proper encryption like AES
    return hashlib.sha256((data + key).encode()).hexdigest()

def validate_api_key(key: str) -> bool:
    """Validate an API key format"""
    # Basic validation - in production, you'd want more thorough checks
    return len(key) >= 32 and key.isalnum()

def sanitize_input(input_str: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    # Remove potentially dangerous characters
    return ''.join(c for c in input_str if c.isalnum() or c in ['-', '_', '.'])

# Initialize components when bot starts
@bot.event
async def setup_hook():
    # Initialize trading bot first
    bot.trading_bot = TradingBot()
    await bot.trading_bot.init_session()
    
    # Then initialize exchange_api with the session
    bot.trading_bot.exchange_api = ExchangeAPI(bot.trading_bot.session)
    
    # Initialize other components
    bot.trading_bot.portfolio_manager = PortfolioManager()
    bot.trading_bot.trading_signals = TradingSignals()

# Clean up when bot closes
@bot.event
async def close():
    await bot.trading_bot.close()

async def cleanup():
    if hasattr(bot, 'bot.trading_bot'):
        await bot.trading_bot.close()
        
@atexit.register
def sync_cleanup():
    # This runs when the script exits, even on error
    loop = asyncio.get_event_loop()
    loop.run_until_complete(cleanup())

# Run the bot
if __name__ == '__main__':
    # Load configuration
    if not os.path.exists('config.json'):
        with open('config.json', 'w') as f:
            json.dump({
                "discord_token": "YOUR_DISCORD_BOT_TOKEN",
                "prefix": "!",
                "admin_ids": []
            }, f, indent=2)
        print("Created config.json. Please fill in your Discord bot token.")
        exit()
    
    try:
        bot.run(config['discord_token'])
    except discord.LoginFailure:
        print("Invalid Discord token. Please check your config.json file.")
    except Exception as e:
        print(f"Error starting bot: {e}")