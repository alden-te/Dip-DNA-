import pandas as pd
import numpy as np
from datetime import datetime

class ForensicAnalyzer:
    def __init__(self, indicator_calc, pivot_detector):
        self.indicator_calc = indicator_calc
        self.pivot_detector = pivot_detector
    
    def analyze_single_bar(self, df, bar_idx):
        if bar_idx < 0 or bar_idx >= len(df):
            return None
        
        row = df.iloc[bar_idx]
        date = df.index[bar_idx]
        
        close = float(row['Close'])
        open_p = float(row['Open'])
        high = float(row['High'])
        low = float(row['Low'])
        volume = float(row['Volume'])
        
        body = abs(close - open_p)
        upper_wick = high - max(close, open_p)
        lower_wick = min(close, open_p) - low
        total_range = high - low if high > low else 0.0001
        
        is_doji = 1 if (body / total_range < 0.1) else 0
        is_hammer = 1 if (lower_wick > body * 2 and upper_wick < body * 0.5) else 0
        is_shooting_star = 1 if (upper_wick > body * 2 and lower_wick < body * 0.5) else 0
        wick_ratio = lower_wick / body if body > 0 else 0
        
        ma_info = self.indicator_calc.calculate_ma_tangle(df, bar_idx)
        
        indicators = {
            'rsi_14': float(row.get('RSI_14', np.nan)),
            'rsi_7': float(row.get('RSI_7', np.nan)),
            'mfi_14': float(row.get('MFI_14', np.nan)),
            'stochrsi_k': float(row.get('STOCHRSIk_14_14_3_3', np.nan)),
            'stochrsi_d': float(row.get('STOCHRSId_14_14_3_3', np.nan)),
            'willr_14': float(row.get('WILLR_14', np.nan)),
            'cci_20': float(row.get('CCI_20_20', np.nan)),
            'macd': float(row.get('MACD_12_26_9', np.nan)),
            'macd_signal': float(row.get('MACDs_12_26_9', np.nan)),
            'macd_hist': float(row.get('MACDh_12_26_9', np.nan)),
            'bbp_20': float(row.get('BBP_20_2.0', np.nan)),
            'bb_width_20': float(row.get('BBB_20_2.0', np.nan)),
            'atr_14': float(row.get('ATR_14', np.nan)),
            'adx_14': float(row.get('ADX_14', np.nan)),
            'volume_ratio_20': float(row.get('Volume_Ratio_20', np.nan)),
        }
        
        if bar_idx >= 20:
            lowest_20 = df['Low'].iloc[bar_idx-19:bar_idx+1].min()
            highest_20 = df['High'].iloc[bar_idx-19:bar_idx+1].max()
            range_20 = highest_20 - lowest_20
            price_position = ((close - lowest_20) / range_20) * 100 if range_20 > 0 else 50
        else:
            price_position = 50
        
        if bar_idx >= 100:
            swing_high = df['High'].iloc[bar_idx-100:bar_idx+1].max()
            swing_low = df['Low'].iloc[bar_idx-100:bar_idx+1].min()
            fib_diff = swing_high - swing_low
            if fib_diff > 0:
                fib_level = (swing_high - close) / fib_diff
            else:
                fib_level = 0.5
        else:
            fib_level = 0.5
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'close': close,
            'body': body,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'wick_ratio': wick_ratio,
            'is_doji': is_doji,
            'is_hammer': is_hammer,
            'is_shooting_star': is_shooting_star,
            'ema_tangle': ma_info['ema_tangle'],
            'sma_tangle': ma_info['sma_tangle'],
            'bearish_alignment': ma_info['bearish_alignment'],
            'price_position': price_position,
            'fib_level': fib_level,
            **indicators
        }
    
    def analyze_dip_with_history(self, df, dip_idx, lookback=100, lookahead=60):
        if dip_idx < lookback or dip_idx >= len(df) - lookahead:
            return None
        
        pre_dip_bars = []
        for i in range(dip_idx - lookback, dip_idx):
            bar_data = self.analyze_single_bar(df, i)
            if bar_data:
                pre_dip_bars.append(bar_data)
        
        dip_bar = self.analyze_single_bar(df, dip_idx)
        
        post_dip_bars = []
        for i in range(dip_idx + 1, dip_idx + lookahead + 1):
            bar_data = self.analyze_single_bar(df, i)
            if bar_data:
                post_dip_bars.append(bar_data)
        
        pre_stats = self._calculate_window_stats(pre_dip_bars)
        
        dip_price = dip_bar['close']
        post_stats = self._calculate_post_performance(post_dip_bars, dip_price)
        
        return {
            'dip_bar': dip_bar,
            'pre_stats': pre_stats,
            'post_stats': post_stats,
            'all_pre_bars': pre_dip_bars,
            'all_post_bars': post_dip_bars
        }
    
    def _calculate_window_stats(self, bars):
        if not bars:
            return {}
        
        df_bars = pd.DataFrame(bars)
        
        stats = {
            'rsi_mean': df_bars['rsi_14'].mean(),
            'rsi_std': df_bars['rsi_14'].std(),
            'rsi_min': df_bars['rsi_14'].min(),
            'rsi_max': df_bars['rsi_14'].max(),
            'rsi_below_30_count': int((df_bars['rsi_14'] < 30).sum()),
            'mfi_mean': df_bars['mfi_14'].mean(),
            'mfi_below_30_count': int((df_bars['mfi_14'] < 30).sum()),
            'ema_tangle_mean': df_bars['ema_tangle'].mean(),
            'ema_tangle_std': df_bars['ema_tangle'].std(),
            'volume_ratio_mean': df_bars['volume_ratio_20'].mean(),
            'volume_spike_count': int((df_bars['volume_ratio_20'] > 2.0).sum()),
            'bearish_days': int((df_bars['bearish_alignment'] == 1).sum()),
            'hammer_count': int(df_bars['is_hammer'].sum()),
            'doji_count': int(df_bars['is_doji'].sum())
        }
        
        return stats
    
    def _calculate_post_performance(self, post_bars, entry_price):
        if not post_bars:
            return {}
        
        df_post = pd.DataFrame(post_bars)
        
        returns = (df_post['close'] / entry_price - 1) * 100
        
        stats = {
            'ret_5d': returns.iloc[4] if len(returns) >= 5 else np.nan,
            'ret_20d': returns.iloc[19] if len(returns) >= 20 else np.nan,
            'ret_60d': returns.iloc[-1] if len(returns) >= 60 else np.nan,
            'max_profit': returns.max(),
            'max_drawdown': returns.min(),
            'days_to_max_profit': int(returns.argmax()),
            'days_in_profit': int((returns > 0).sum()),
            'days_in_loss': int((returns < 0).sum())
        }
        
        return stats
