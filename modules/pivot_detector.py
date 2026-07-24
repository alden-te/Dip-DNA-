import numpy as np
import pandas as pd

class PivotDetector:
    """80-80 pivot ve diğer pivot tespiti"""
    
    def __init__(self, left_bars=80, right_bars=80):
        self.left_bars = left_bars
        self.right_bars = right_bars
    
    def find_80_80_pivot_lows(self, df):
        """
        TradingView mantığıyla 80-80 pivot low tespiti
        İlk ve son 80 bar için esnek kontrol
        """
        if df is None or len(df) < (self.left_bars + self.right_bars + 1):
            return []
        
        lows = df['Low'].values
        n = len(lows)
        pivot_indices = []
        
        for i in range(n):
            # Sol pencere (başlangıçta daha küçük olabilir)
            left_start = max(0, i - self.left_bars)
            left_window = lows[left_start:i] if i > left_start else np.array([])
            
            # Sağ pencere (sonda daha küçük olabilir)
            right_end = min(n, i + self.right_bars + 1)
            right_window = lows[i+1:right_end] if i+1 < right_end else np.array([])
            
            current_low = lows[i]
            
            # Sol tarafta en düşük mü?
            is_lowest_left = len(left_window) == 0 or np.all(current_low <= left_window)
            
            # Sağ tarafta en düşük mü?
            is_lowest_right = len(right_window) == 0 or np.all(current_low <= right_window)
            
            # Her iki tarafta da en düşükse pivot
            if is_lowest_left and is_lowest_right:
                # En az bir tarafta yeterli bar olmalı
                left_count = i - left_start
                right_count = right_end - (i + 1)
                
                if left_count >= self.left_bars or right_count >= self.right_bars or (left_count > 0 and right_count > 0):
                    pivot_indices.append(i)
        
        return pivot_indices
    
    def find_zigzag_dips(self, df, threshold=0.15):
        """
        ZigZag algoritması ile dip tespiti
        threshold: %15 için 0.15
        """
        if df is None or len(df) < 100:
            return []
        
        prices = df['Close'].values
        n = len(prices)
        dips = []
        
        i = 0
        while i < n - 1:
            # Zirve bul
            if i > 0:
                # Yükseliş trendi
                if prices[i] > prices[i-1]:
                    high_idx = i
                    while i < n - 1 and prices[i] >= prices[high_idx]:
                        if prices[i] > prices[high_idx]:
                            high_idx = i
                        i += 1
                    
                    # Sonra düşüş ara
                    if i < n - 1:
                        low_idx = i
                        while i < n - 1 and prices[i] <= prices[low_idx]:
                            if prices[i] < prices[low_idx]:
                                low_idx = i
                            i += 1
                        
                        # Zirve-dip farkı threshold'dan fazla mı?
                        if high_idx < low_idx:
                            drop_pct = (prices[low_idx] / prices[high_idx] - 1)
                            if drop_pct <= -threshold:
                                dips.append(low_idx)
            
            i += 1
        
        return dips
    
    def find_local_minimums(self, df, window=20):
        """Yerel minimumları bul (son window barın en düşüğü)"""
        if df is None or len(df) < window:
            return []
        
        lows = df['Low'].values
        n = len(lows)
        minimums = []
        
        for i in range(window, n):
            window_lows = lows[i-window:i+1]
            if lows[i] == window_lows.min():
                minimums.append(i)
        
        return minimums
