import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tradingview_screener import get_all_symbols
import time
from tqdm import tqdm

class DataCollector:
    """Profesyonel veri toplama sınıfı"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(minutes=30)
    
    def get_all_bist_tickers(self):
        """Tüm BIST hisselerini getir"""
        try:
            symbols = get_all_symbols(market='turkey')
            tickers = [s.split(':')[1] + '.IS' for s in symbols if 'BIST:' in s]
            return tickers
        except Exception as e:
            print(f"Hata: {e}")
            return ["THYAO.IS", "ASELS.IS", "GARAN.IS", "EREGL.IS", "SISE.IS"]
    
    def download_stock_data(self, ticker, period="10y", start_date=None, end_date=None):
        """Hisse verisini indir ve temizle"""
        cache_key = f"{ticker}_{period}_{start_date}_{end_date}"
        
        # Cache kontrolü
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
        
        try:
            if start_date and end_date:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=15)
            else:
                df = yf.download(ticker, period=period, progress=False, timeout=15)
            
            if df.empty or len(df) < 100:
                return None
            
            # MultiIndex temizliği
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Volume ve OHLC düzeltme
            if isinstance(df['Volume'], pd.DataFrame):
                df['Volume'] = df['Volume'].iloc[:, 0]
            
            for col in ['Open', 'High', 'Low', 'Close']:
                if isinstance(df[col], pd.DataFrame):
                    df[col] = df[col].iloc[:, 0]
            
            df = df.ffill().dropna()
            
            if len(df) < 100:
                return None
            
            # Cache'e kaydet
            self.cache[cache_key] = (df, datetime.now())
            
            return df
            
        except Exception as e:
            print(f"{ticker} hatası: {e}")
            return None
    
    def download_multiple_stocks(self, tickers, period="10y", show_progress=True):
        """Çoklu hisse indir"""
        all_data = {}
        
        iterator = tqdm(tickers, desc="Veri indiriliyor") if show_progress else tickers
        
        for ticker in iterator:
            df = self.download_stock_data(ticker, period)
            if df is not None:
                all_data[ticker] = df
        
        return all_data
