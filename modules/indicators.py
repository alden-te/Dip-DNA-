import pandas as pd
import numpy as np

class IndicatorCalculator:
    def __init__(self):
        self.ema_periods = [5, 8, 10, 13, 15, 20, 21, 25, 30, 34, 35, 40, 45, 50, 
                           55, 60, 70, 80, 89, 90, 100, 110, 120, 144, 150, 200, 233, 250, 300, 350, 377, 400, 500, 600, 800]
        self.sma_periods = [5, 8, 10, 13, 15, 20, 21, 25, 30, 34, 35, 40, 45, 50, 
                           55, 60, 70, 80, 89, 90, 100, 110, 120, 144, 150, 200, 233, 250, 300, 350, 377, 400, 500, 600, 800]
    
    @staticmethod
    def _rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _mfi(high, low, close, volume, period=14):
        tp = (high + low + close) / 3
        mf = tp * volume
        delta = tp.diff()
        pos = mf.where(delta > 0, 0).rolling(window=period).sum()
        neg = mf.where(delta < 0, 0).rolling(window=period).sum()
        return 100 - (100 / (1 + pos / neg))
    
    @staticmethod
    def _stoch_rsi(rsi, period=14, k=3, d=3):
        low_rsi = rsi.rolling(window=period).min()
        high_rsi = rsi.rolling(window=period).max()
        stoch = 100 * ((rsi - low_rsi) / (high_rsi - low_rsi))
        k_line = stoch.rolling(window=k).mean()
        d_line = k_line.rolling(window=d).mean()
        return k_line, d_line
    
    @staticmethod
    def _williams_r(high, low, close, period=14):
        hh = high.rolling(window=period).max()
        ll = low.rolling(window=period).min()
        return -100 * ((hh - close) / (hh - ll))
    
    @staticmethod
    def _cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad)
    
    @staticmethod
    def _macd(close, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist
    
    @staticmethod
    def _bbands(close, period=20, std_dev=2):
        mid = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = mid + (std_dev * std)
        lower = mid - (std_dev * std)
        pct = (close - lower) / (upper - lower)
        width = ((upper - lower) / mid) * 100
        return upper, mid, lower, pct, width
    
    @staticmethod
    def _atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def _adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()
    
    @staticmethod
    def _obv(close, volume):
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=close.index)
    
    def calculate_all_indicators(self, df):
        if df is None or len(df) < 100:
            return None
        df = df.copy()
        
        for p in self.ema_periods:
            try:
                df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
                df[f'Dist_EMA_{p}_Pct'] = ((df['Close'] - df[f'EMA_{p}']) / df[f'EMA_{p}']) * 100
            except: pass
        
        for p in self.sma_periods:
            try:
                df[f'SMA_{p}'] = df['Close'].rolling(window=p).mean()
                df[f'Dist_SMA_{p}_Pct'] = ((df['Close'] - df[f'SMA_{p}']) / df[f'SMA_{p}']) * 100
            except: pass
        
        try:
            df['RSI_14'] = self._rsi(df['Close'], 14)
            df['RSI_7'] = self._rsi(df['Close'], 7)
            df['RSI_21'] = self._rsi(df['Close'], 21)
        except: pass
        
        try:
            df['MFI_14'] = self._mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)
            df['MFI_7'] = self._mfi(df['High'], df['Low'], df['Close'], df['Volume'], 7)
        except: pass
        
        try:
            k, d = self._stoch_rsi(df['RSI_14'], 14, 3, 3)
            df['STOCHRSIk_14_14_3_3'] = k
            df['STOCHRSId_14_14_3_3'] = d
        except: pass
        
        try:
            df['WILLR_14'] = self._williams_r(df['High'], df['Low'], df['Close'], 14)
            df['WILLR_7'] = self._williams_r(df['High'], df['Low'], df['Close'], 7)
        except: pass
        
        try:
            df['CCI_20_20'] = self._cci(df['High'], df['Low'], df['Close'], 20)
            df['CCI_14_14'] = self._cci(df['High'], df['Low'], df['Close'], 14)
        except: pass
        
        try:
            m, s, h = self._macd(df['Close'], 12, 26, 9)
            df['MACD_12_26_9'] = m
            df['MACDs_12_26_9'] = s
            df['MACDh_12_26_9'] = h
        except: pass
        
        try:
            u, m, l, p, w = self._bbands(df['Close'], 20, 2)
            df['BBU_20_2.0'] = u
            df['BBM_20_2.0'] = m
            df['BBL_20_2.0'] = l
            df['BBP_20_2.0'] = p
            df['BBB_20_2.0'] = w
        except: pass
        
        try:
            df['ATR_14'] = self._atr(df['High'], df['Low'], df['Close'], 14)
            df['ATR_7'] = self._atr(df['High'], df['Low'], df['Close'], 7)
        except: pass
        
        try:
            df['ADX_14'] = self._adx(df['High'], df['Low'], df['Close'], 14)
        except: pass
        
        try:
            df['OBV'] = self._obv(df['Close'], df['Volume'])
        except: pass
        
        try:
            df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
            df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
            df['Volume_SMA_50'] = df['Volume'].rolling(50).mean()
            df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
        except: pass
        
        try:
            df['Return_1D'] = df['Close'].pct_change() * 100
            df['Return_5D'] = df['Close'].pct_change(5) * 100
            df['Return_20D'] = df['Close'].pct_change(20) * 100
            df['Return_60D'] = df['Close'].pct_change(60) * 100
        except: pass
        
        try:
            df['Volatility_20D'] = df['Return_1D'].rolling(20).std() * np.sqrt(252) * 100
        except: pass
        
        return df
    
    def calculate_ma_tangle(self, df, idx):
        row = df.iloc[idx]
        ema_values = []
        for p in [8, 13, 21, 50, 89, 200]:
            col = f'EMA_{p}'
            if col in df.columns:
                val = row.get(col, np.nan)
                if pd.notna(val) and val > 0:
                    ema_values.append(val)
        ema_tangle = (np.std(ema_values) / np.mean(ema_values)) * 100 if len(ema_values) > 1 else 50
        
        sma_values = []
        for p in [20, 50, 100, 200]:
            col = f'SMA_{p}'
            if col in df.columns:
                val = row.get(col, np.nan)
                if pd.notna(val) and val > 0:
                    sma_values.append(val)
        sma_tangle = (np.std(sma_values) / np.mean(sma_values)) * 100 if len(sma_values) > 1 else 50
        
        bearish_alignment = 0
        if len(ema_values) >= 3:
            if all(ema_values[i] > ema_values[i+1] for i in range(len(ema_values)-1)):
                bearish_alignment = 1
            elif all(ema_values[i] < ema_values[i+1] for i in range(len(ema_values)-1)):
                bearish_alignment = -1
        
        return {
            'ema_tangle': ema_tangle,
            'sma_tangle': sma_tangle,
            'bearish_alignment': bearish_alignment,
            'ema_count': len(ema_values),
            'sma_count': len(sma_values)
        }
