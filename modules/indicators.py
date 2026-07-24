import numpy as np

class ManualIndicators:
    """pandas-ta olmadan tüm indikatörleri manuel hesaplar"""
    
    @staticmethod
    def ema(series, period):
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series, period):
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def rsi(series, period=14):
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def mfi(high, low, close, volume, period=14):
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0)
        negative_flow = money_flow.where(delta < 0, 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    @staticmethod
    def stoch_rsi(rsi, period=14, smooth_k=3, smooth_d=3):
        """Stochastic RSI"""
        lowest_rsi = rsi.rolling(window=period).min()
        highest_rsi = rsi.rolling(window=period).max()
        
        stoch_rsi = 100 * ((rsi - lowest_rsi) / (highest_rsi - lowest_rsi))
        k = stoch_rsi.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        
        return k, d
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        willr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return willr
    
    @staticmethod
    def cci(high, low, close, period=20):
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        """MACD"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(close, period=20, std_dev=2):
        """Bollinger Bands"""
        middle_band = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        # %B indicator
        bb_pct = (close - lower_band) / (upper_band - lower_band)
        
        # Bandwidth
        bb_width = ((upper_band - lower_band) / middle_band) * 100
        
        return upper_band, middle_band, lower_band, bb_pct, bb_width
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def adx(high, low, close, period=14):
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = true_range.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def obv(close, volume):
        """On Balance Volume"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=close.index)

class IndicatorCalculator:
    """Tüm indikatörleri hesaplayan ana sınıf"""
    
    def __init__(self):
        self.manual = ManualIndicators()
        
        # EMA periyotları
        self.ema_periods = [5, 8, 10, 13, 15, 20, 21, 25, 30, 34, 35, 40, 45, 50, 
                           55, 60, 70, 80, 89, 90, 100, 110, 120, 144, 150, 200, 233, 250, 300, 350, 377, 400, 500, 600, 800]
        
        # SMA periyotları
        self.sma_periods = [5, 8, 10, 13, 15, 20, 21, 25, 30, 34, 35, 40, 45, 50, 
                           55, 60, 70, 80, 89, 90, 100, 110, 120, 144, 150, 200, 233, 250, 300, 350, 377, 400, 500, 600, 800]
    
    def calculate_all_indicators(self, df):
        """Tüm indikatörleri hesapla"""
        if df is None or len(df) < 100:
            return None
        
        df = df.copy()
        
        # EMA'lar
        for period in self.ema_periods:
            try:
                df[f'EMA_{period}'] = self.manual.ema(df['Close'], period)
                df[f'Dist_EMA_{period}_Pct'] = ((df['Close'] - df[f'EMA_{period}']) / df[f'EMA_{period}']) * 100
            except:
                pass
        
        # SMA'lar
        for period in self.sma_periods:
            try:
                df[f'SMA_{period}'] = self.manual.sma(df['Close'], period)
                df[f'Dist_SMA_{period}_Pct'] = ((df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']) * 100
            except:
                pass
        
        # RSI
        try:
            df['RSI_14'] = self.manual.rsi(df['Close'], 14)
            df['RSI_7'] = self.manual.rsi(df['Close'], 7)
            df['RSI_21'] = self.manual.rsi(df['Close'], 21)
        except:
            pass
        
        # MFI
        try:
            df['MFI_14'] = self.manual.mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)
            df['MFI_7'] = self.manual.mfi(df['High'], df['Low'], df['Close'], df['Volume'], 7)
        except:
            pass
        
        # StochRSI
        try:
            rsi_14 = df['RSI_14']
            stoch_k, stoch_d = self.manual.stoch_rsi(rsi_14, 14, 3, 3)
            df['STOCHRSIk_14_14_3_3'] = stoch_k
            df['STOCHRSId_14_14_3_3'] = stoch_d
        except:
            pass
        
        # Williams %R
        try:
            df['WILLR_14'] = self.manual.williams_r(df['High'], df['Low'], df['Close'], 14)
            df['WILLR_7'] = self.manual.williams_r(df['High'], df['Low'], df['Close'], 7)
        except:
            pass
        
        # CCI
        try:
            df['CCI_20_20'] = self.manual.cci(df['High'], df['Low'], df['Close'], 20)
            df['CCI_14_14'] = self.manual.cci(df['High'], df['Low'], df['Close'], 14)
        except:
            pass
        
        # MACD
        try:
            macd, signal, hist = self.manual.macd(df['Close'], 12, 26, 9)
            df['MACD_12_26_9'] = macd
            df['MACDs_12_26_9'] = signal
            df['MACDh_12_26_9'] = hist
        except:
            pass
        
        # Bollinger Bands
        try:
            upper, middle, lower, bb_pct, bb_width = self.manual.bollinger_bands(df['Close'], 20, 2)
            df['BBU_20_2.0'] = upper
            df['BBM_20_2.0'] = middle
            df['BBL_20_2.0'] = lower
            df['BBP_20_2.0'] = bb_pct
            df['BBB_20_2.0'] = bb_width
        except:
            pass
        
        # ATR
        try:
            df['ATR_14'] = self.manual.atr(df['High'], df['Low'], df['Close'], 14)
            df['ATR_7'] = self.manual.atr(df['High'], df['Low'], df['Close'], 7)
        except:
            pass
        
        # ADX
        try:
            adx, plus_di, minus_di = self.manual.adx(df['High'], df['Low'], df['Close'], 14)
            df['ADX_14'] = adx
        except:
            pass
        
        # OBV
        try:
            df['OBV'] = self.manual.obv(df['Close'], df['Volume'])
        except:
            pass
        
        # Hacim SMA
        try:
            df['Volume_SMA_10'] = self.manual.sma(df['Volume'], 10)
            df['Volume_SMA_20'] = self.manual.sma(df['Volume'], 20)
            df['Volume_SMA_50'] = self.manual.sma(df['Volume'], 50)
            df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
        except:
            pass
        
        # Fiyat değişimleri
        try:
            df['Return_1D'] = df['Close'].pct_change() * 100
            df['Return_5D'] = df['Close'].pct_change(5) * 100
            df['Return_20D'] = df['Close'].pct_change(20) * 100
            df['Return_60D'] = df['Close'].pct_change(60) * 100
        except:
            pass
        
        # Volatilite
        try:
            df['Volatility_20D'] = df['Return_1D'].rolling(20).std() * np.sqrt(252) * 100
        except:
            pass
        
        return df
    
    def calculate_ma_tangle(self, df, idx):
        """MA sıkışma (tangle) hesapla"""
        row = df.iloc[idx]
        
        # EMA tangle
        ema_values = []
        for period in [8, 13, 21, 50, 89, 200]:
            col = f'EMA_{period}'
            if col in df.columns:
                val = row.get(col, np.nan)
                if pd.notna(val) and val > 0:
                    ema_values.append(val)
        
        ema_tangle = (np.std(ema_values) / np.mean(ema_values)) * 100 if len(ema_values) > 1 else 50
        
        # SMA tangle
        sma_values = []
        for period in [20, 50, 100, 200]:
            col = f'SMA_{period}'
            if col in df.columns:
                val = row.get(col, np.nan)
                if pd.notna(val) and val > 0:
                    sma_values.append(val)
        
        sma_tangle = (np.std(sma_values) / np.mean(sma_values)) * 100 if len(sma_values) > 1 else 50
        
        # MA hizalanma (alignment)
        bearish_alignment = 0
        if len(ema_values) >= 3:
            if all(ema_values[i] > ema_values[i+1] for i in range(len(ema_values)-1)):
                bearish_alignment = 1  # Bearish (fiyat düşüyor)
            elif all(ema_values[i] < ema_values[i+1] for i in range(len(ema_values)-1)):
                bearish_alignment = -1  # Bullish (fiyat yükseliyor)
        
        return {
            'ema_tangle': ema_tangle,
            'sma_tangle': sma_tangle,
            'bearish_alignment': bearish_alignment,
            'ema_count': len(ema_values),
            'sma_count': len(sma_values)
        }
