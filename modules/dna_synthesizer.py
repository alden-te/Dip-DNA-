import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

class DNASynthesizer:
    """Dip DNA sentezi ve istatistiksel analiz"""
    
    def __init__(self):
        pass
    
    def create_stock_dna(self, dip_analyses):
        """
        Bir hissenin tüm dip analizlerinden DNA çıkar
        İstatistiksel özet, medyanlar, yüzdelikler
        """
        if not dip_analyses:
            return None
        
        # Başarılı ve başarısız dipleri ayır
        successful = [d for d in dip_analyses if d['post_stats'].get('ret_20d', 0) >= 10.0]
        failed = [d for d in dip_analyses if d['post_stats'].get('ret_20d', 0) < 10.0 or pd.isna(d['post_stats'].get('ret_20d', np.nan))]
        
        # Tüm dipler için DNA
        dna = self._create_dna_from_list(dip_analyses, "all")
        
        # Başarılı dipler için DNA
        if len(successful) >= 5:
            dna_successful = self._create_dna_from_list(successful, "successful")
        else:
            dna_successful = None
        
        # Başarısız dipler için DNA
        if len(failed) >= 5:
            dna_failed = self._create_dna_from_list(failed, "failed")
        else:
            dna_failed = None
        
        # İstatistiksel anlamlılık testi
        significance = self._test_significance(successful, failed) if len(successful) >= 5 and len(failed) >= 5 else None
        
        return {
            'all': dna,
            'successful': dna_successful,
            'failed': dna_failed,
            'significance': significance,
            'total_dips': len(dip_analyses),
            'successful_dips': len(successful),
            'failed_dips': len(failed),
            'success_rate': len(successful) / len(dip_analyses) * 100 if dip_analyses else 0
        }
    
    def _create_dna_from_list(self, dip_analyses, dna_type):
        """Bir dip listesinden DNA oluştur"""
        if not dip_analyses:
            return None
        
        # Tüm dip barları topla
        dip_bars = [d['dip_bar'] for d in dip_analyses]
        pre_stats_list = [d['pre_stats'] for d in dip_analyses]
        
        df_bars = pd.DataFrame(dip_bars)
        df_pre = pd.DataFrame(pre_stats_list)
        
        # Temel DNA
        dna = {
            'type': dna_type,
            'count': len(dip_analyses),
            
            # Dip anı değerleri (medyan)
            'dip_rsi_median': float(df_bars['rsi_14'].median()),
            'dip_rsi_mean': float(df_bars['rsi_14'].mean()),
            'dip_rsi_std': float(df_bars['rsi_14'].std()),
            'dip_mfi_median': float(df_bars['mfi_14'].median()),
            'dip_stochrsi_median': float(df_bars['stochrsi_k'].median()),
            'dip_willr_median': float(df_bars['willr_14'].median()),
            'dip_ema_tangle_median': float(df_bars['ema_tangle'].median()),
            'dip_bbp_median': float(df_bars['bbp_20'].median()),
            'dip_volume_ratio_median': float(df_bars['volume_ratio_20'].median()),
            'dip_price_position_median': float(df_bars['price_position'].median()),
            'dip_fib_level_median': float(df_bars['fib_level'].median()),
            'dip_hammer_ratio': float(df_bars['is_hammer'].mean()),
            
            # 100 bar öncesi istatistikler
            'pre_rsi_mean': float(df_pre['rsi_mean'].mean()),
            'pre_rsi_below_30_avg': float(df_pre['rsi_below_30_count'].mean()),
            'pre_mfi_mean': float(df_pre['mfi_mean'].mean()),
            'pre_ema_tangle_mean': float(df_pre['ema_tangle_mean'].mean()),
            'pre_volume_spike_avg': float(df_pre['volume_spike_count'].mean()),
            'pre_bearish_days_avg': float(df_pre['bearish_days'].mean()),
            
            # Sonrası performans
            'avg_ret_5d': float(df_bars.apply(lambda x: x.get('post_stats', {}).get('ret_5d', np.nan), axis=1).mean()),
            'avg_ret_20d': float(df_bars.apply(lambda x: x.get('post_stats', {}).get('ret_20d', np.nan), axis=1).mean()),
            'avg_max_profit': float(df_bars.apply(lambda x: x.get('post_stats', {}).get('max_profit', np.nan), axis=1).mean()),
            'avg_max_drawdown': float(df_bars.apply(lambda x: x.get('post_stats', {}).get('max_drawdown', np.nan), axis=1).mean()),
        }
        
        # Yüzdelikler (quantiles)
        for col in ['rsi_14', 'mfi_14', 'ema_tangle', 'volume_ratio_20', 'price_position']:
            if col in df_bars.columns:
                dna[f'{col}_25pct'] = float(df_bars[col].quantile(0.25))
                dna[f'{col}_75pct'] = float(df_bars[col].quantile(0.75))
        
        return dna
    
    def _test_significance(self, successful, failed):
        """Başarılı ve başarısız dipler arası istatistiksel anlamlılık testi"""
        if len(successful) < 5 or len(failed) < 5:
            return None
        
        significance = {}
        
        # Test edilecek metrikler
        metrics = ['rsi_14', 'mfi_14', 'ema_tangle', 'volume_ratio_20', 'price_position', 'fib_level']
        
        for metric in metrics:
            try:
                succ_vals = [d['dip_bar'][metric] for d in successful if pd.notna(d['dip_bar'].get(metric))]
                fail_vals = [d['dip_bar'][metric] for d in failed if pd.notna(d['dip_bar'].get(metric))]
                
                if len(succ_vals) >= 5 and len(fail_vals) >= 5:
                    t_stat, p_value = scipy_stats.ttest_ind(succ_vals, fail_vals)
                    significance[metric] = {
                        't_stat': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'succ_mean': float(np.mean(succ_vals)),
                        'fail_mean': float(np.mean(fail_vals))
                    }
            except:
                continue
        
        return significance
    
    def compare_dnas(self, dna1, dna2, name1="DNA1", name2="DNA2"):
        """İki DNA'yı karşılaştır"""
        if not dna1 or not dna2:
            return None
        
        comparison = {
            'name1': name1,
            'name2': name2,
            'differences': {}
        }
        
        # Karşılaştırılacak metrikler
        metrics = ['dip_rsi_median', 'dip_mfi_median', 'dip_ema_tangle_median', 
                   'dip_volume_ratio_median', 'dip_price_position_median']
        
        for metric in metrics:
            if metric in dna1 and metric in dna2:
                diff = dna1[metric] - dna2[metric]
                pct_diff = (diff / dna2[metric] * 100) if dna2[metric] != 0 else 0
                comparison['differences'][metric] = {
                    'val1': dna1[metric],
                    'val2': dna2[metric],
                    'diff': diff,
                    'pct_diff': pct_diff
                }
        
        return comparison
