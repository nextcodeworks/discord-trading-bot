# Discord Trading Bot  

**A sophisticated cryptocurrency trading bot for Discord** with real-time market data, trading signals, portfolio management, paper trading, and backtesting capabilities.  

⭐ **Leave a star on [GitHub](https://github.com/nextcodeworks/DiscordTradingBot) if you like this project!** ⭐  

---

## 📸 Screenshots  

*(Space for screenshots of the bot in action - add images showing command examples, portfolio views, trading signals, etc.)*  

---

## 🛠 Installation  

### 1️⃣ Install TA-Lib (Technical Analysis Library)  
Before installing Python dependencies, you need to install **TA-Lib**, which is used for technical indicators.  

#### **Windows:**  
1. Download the TA-Lib installer from [this link](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)  
2. Run the installer  
3. Set these environment variables (replace `C:\TA-Lib` with your install path):  
   - **TA_LIBRARY_PATH** = `C:\TA-Lib\lib\ta_lib.dll`  
   - **TA_INCLUDE_PATH** = `C:\TA-Lib\include`  
   - **TA_BIN_PATH** = `C:\TA-Lib\bin`  

#### **Linux/macOS:**  
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

### 2️⃣ Install Python Dependencies  
Run:  
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure the Bot  
1. Open `config.json` and add your **Discord Bot Token**:  
   ```json
   {
       "discord_token": "YOUR_BOT_TOKEN_HERE",
       "prefix": "/",
       "admin_ids": []
   }
   ```
2. Save the file.  

### 4️⃣ Run the Bot  
```bash
python main.py
```

### 5️⃣ Invite the Bot to Your Server  
- Generate an invite link with **bot** and **messages** permissions.  
- Join a server and start trading!  

---

## 📜 Commands & Examples  

### 📊 **Market Data**  
- `/price BTCUSDT` → Get BTC price  
- `/market ETHUSDT binance` → Get detailed ETH market data  

### 📈 **Trading Signals**  
- `/signals BTCUSDT` → Default signals (1d, 200 candles)  
- `/signals SOLUSDT 1h 50` → 1-hour signals with 50 data points  

### 💰 **Portfolio Management**  
- `/portfolio` → View real portfolio  
- `/portfolio paper` → View paper trading balance  

### 🛒 **Trading**  
- `/buy BTCUSDT 0.01 paper` → Buy 0.01 BTC with paper trading  
- `/sell ETHUSDT 0.5 1800 paper` → Sell 0.5 ETH at $1800 (paper)  

### 📝 **Paper Trading**  
- `/paper reset` → Reset to $10,000 virtual balance  
- `/paper balance` → Show detailed paper portfolio  

### 🧪 **Backtesting**  
- `/backtest BTCUSDT` → Default SMA crossover test  
- `/backtest ADAUSDT sma_crossover 4h 500` → Custom backtest  

### ℹ️ **Help & Info**  
- `/bothelp` → List all commands  
- `/about` → Bot info & links  

---

## 📜 License  
This project is under the **MIT License**.  

## ❓ Support  
For help or suggestions, contact me at:  
- 🌐 [Next Code Works](https://nextcodeworks.github.io/)  
- 💻 [GitHub Profile](https://github.com/nextcodeworks)  

Happy trading! 🚀