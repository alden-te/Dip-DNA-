import pandas as pd
import numpy as np
import pandas_ta as ta

class IndicatorCalculator:
    """40+ indikatör hesaplama sınıfı"""
    
    def __init__(self):
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
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                df[f'Dist_EMA_{period}_Pct'] = ((df['Close'] - df[f'EMA_{period}']) / df[f'EMA_{period}']) * 100
            except:
                pass
        
        # SMA'lar
        for period in self.sma_periods:
            try:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'Dist_SMA_{period}_Pct'] = ((df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}']) * 100
            except:
                pass
        
        # Momentum İndikatörleri
        try:
            df.ta.rsi(length=14, append=True)
            df.ta.rsi(length=7, append=True)
            df.ta.rsi(length=21, append=True)
        except:
            pass
        
        try:
            df.ta.mfi(length=14, append=True)
            df.ta.mfi(length=7, append=True)
        except:
            pass
        
        try:
            df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
        except:
            pass
        
        try:
            df.ta.willr(length=14, append=True)
            df.ta.willr(length=7, append=True)
        except:
            pass
        
        try:
            df.ta.cci(length=20, append=True)
            df.ta.cci(length=14, append=True)
        except:
            pass
        
        # MACD
        try:
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
        except:
            pass
        
        # Bollinger Bands
        try:
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.bbands(length=50, std=2, append=True)
        except:
            pass
        
        # ATR
        try:
            df.ta.atr(length=14, append=True)
            df.ta.atr(length=7, append=True)
        except:
            pass
        
        # ADX
        try:
            df.ta.adx(length=14, append=True)
        except:
            pass
        
        # OBV
        try:
            df.ta.obv(append=True)
        except:
            pass
        
        # Hacim SMA
        try:
            df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
            df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
            df['Volume_SMA_50'] = df['Volume'].rolling(50).mean()
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
