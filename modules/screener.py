import pandas as pd
import numpy as np
from tqdm import tqdm

class StockScreener:
    """Hisse tarama motoru"""
    
    def __init__(self, data_collector, indicator_calc, pivot_detector, forensic_analyzer, dna_synthesizer):
        self.data_collector = data_collector
        self.indicator_calc = indicator_calc
        self.pivot_detector = pivot_detector
        self.forensic_analyzer = forensic_analyzer
        self.dna_synthesizer = dna_synthesizer
    
    def screen_single_stock(self, ticker, period="10y"):
        """Tek bir hisseyi tara ve tüm dip analizlerini yap"""
        # Veri çek
        df = self.data_collector.download_stock_data(ticker, period)
        if df is None:
            return None
        
        # İndikatörleri hesapla
        df = self.indicator_calc.calculate_all_indicators(df)
        if df is None:
            return None
        
        # Pivot dipleri bul
        pivot_indices = self.pivot_detector.find_80_80_pivot_lows(df)
        
        if len(pivot_indices) < 2:
            return None
        
        # Her dip için detaylı analiz
        dip_analyses = []
        for dip_idx in pivot_indices:
            analysis = self.forensic_analyzer.analyze_dip_with_history(df, dip_idx)
            if analysis:
                dip_analyses.append(analysis)
        
        if not dip_analyses:
            return None
        
        # DNA sentezi
        dna = self.dna_synthesizer.create_stock_dna(dip_analyses)
        
        return {
            'ticker': ticker,
            'df': df,
            'pivot_indices': pivot_indices,
            'dip_analyses': dip_analyses,
            'dna': dna
        }
    
    def screen_multiple_stocks(self, tickers, period="10y", show_progress=True):
        """Çoklu hisse tarama"""
        results = {}
        
        iterator = tqdm(tickers, desc="Hisseler taranıyor") if show_progress else tickers
        
        for ticker in iterator:
            try:
                result = self.screen_single_stock(ticker, period)
                if result:
                    results[ticker] = result
            except Exception as e:
                print(f"{ticker} hatası: {e}")
                continue
        
        return results
    
    def find_current_signals(self, ticker, dna, lookback_days=30):
        """
        Güncel veride DNA'ya uyan sinyaller bul
        Son lookback_days içinde
        """
        df = self.data_collector.download_stock_data(ticker, period="3mo")
        if df is None:
            return []
        
        df = self.indicator_calc.calculate_all_indicators(df)
        if df is None or len(df) < lookback_days:
            return []
        
        signals = []
        current_price = float(df.iloc[-1]['Close'])
        
        # Son lookback_days barı kontrol et
        for i in range(len(df)-lookback_days, len(df)):
            bar_data = self.forensic_analyzer.analyze_single_bar(df, i)
            if not bar_data:
                continue
            
            # DNA eşleşme skoru hesapla
            score = self._calculate_dna_match_score(bar_data, dna)
            
            # Filtreler
            if score >= 0.6 and bar_data['rsi_14'] < 45 and bar_data['price_position'] < 30:
                # Fiyat artış kontrolü (sinyalden bugüne)
                signal_price = bar_data['close']
                price_increase = ((current_price / signal_price) - 1) * 100
                
                # %10'dan fazla yükseldiyse atla (fırsat kaçmış)
                if price_increase < 10.0:
                    signals.append({
                        'date': bar_data['date'],
                        'price': signal_price,
                        'current_price': current_price,
                        'price_increase_pct': round(price_increase, 2),
                        'dna_match_score': round(score * 100, 1),
                        'rsi': round(bar_data['rsi_14'], 1),
                        'mfi': round(bar_data['mfi_14'], 1),
                        'ema_tangle': round(bar_data['ema_tangle'], 2),
                        'volume_ratio': round(bar_data['volume_ratio_20'], 2)
                    })
        
        return signals
    
    def _calculate_dna_match_score(self, bar_data, dna):
        """Bar verisinin DNA'ya eşleşme skorunu hesapla"""
        if not dna or not dna.get('all'):
            return 0
        
        dna_all = dna['all']
        score = 0
        max_score = 7
        
        # RSI yakınlığı
        if abs(bar_data['rsi_14'] - dna_all.get('dip_rsi_median', 30)) < 10:
            score += 1
        
        # MFI yakınlığı
        if abs(bar_data['mfi_14'] - dna_all.get('dip_mfi_median', 30)) < 15:
            score += 1
        
        # EMA tangle yakınlığı
        if abs(bar_data['ema_tangle'] - dna_all.get('dip_ema_tangle_median', 10)) < 10:
            score += 1
        
        # Volume ratio yakınlığı
        if abs(bar_data['volume_ratio_20'] - dna_all.get('dip_volume_ratio_median', 1.5)) < 1:
            score += 1
        
        # Price position yakınlığı
        if abs(bar_data['price_position'] - dna_all.get('dip_price_position_median', 20)) < 15:
            score += 1
        
        # RSI < 45
        if bar_data['rsi_14'] < 45:
            score += 1
        
        # Price position < 30
        if bar_data['price_position'] < 30:
            score += 1
        
        return score / max_score
