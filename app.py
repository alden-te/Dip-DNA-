import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from modules.data_collector import DataCollector
from modules.indicators import IndicatorCalculator
from modules.pivot_detector import PivotDetector
from modules.forensic_analyzer import ForensicAnalyzer
from modules.dna_synthesizer import DNASynthesizer
from modules.screener import StockScreener

st.set_page_config(
    page_title="Ultra Pro Dip Analiz Sistemi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

if 'data_collector' not in st.session_state:
    st.session_state.data_collector = DataCollector()
if 'indicator_calc' not in st.session_state:
    st.session_state.indicator_calc = IndicatorCalculator()
if 'pivot_detector' not in st.session_state:
    st.session_state.pivot_detector = PivotDetector(left_bars=80, right_bars=80)
if 'forensic_analyzer' not in st.session_state:
    st.session_state.forensic_analyzer = ForensicAnalyzer(
        st.session_state.indicator_calc,
        st.session_state.pivot_detector
    )
if 'dna_synthesizer' not in st.session_state:
    st.session_state.dna_synthesizer = DNASynthesizer()
if 'screener' not in st.session_state:
    st.session_state.screener = StockScreener(
        st.session_state.data_collector,
        st.session_state.indicator_calc,
        st.session_state.pivot_detector,
        st.session_state.forensic_analyzer,
        st.session_state.dna_synthesizer
    )

st.markdown('<div class="main-header"> ULTRA PRO DİP ANALİZ SİSTEMİ</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    analysis_mode = st.radio(
        "Analiz Modu",
        ["Tek Hisse Analizi", "Çoklu Hisse Tarama", "Tüm BIST Tarama"],
        index=0
    )
    
    st.markdown("---")
    
    if analysis_mode == "Tek Hisse Analizi":
        ticker = st.text_input("Hisse Kodu", "THYAO.IS").upper()
        if not ticker.endswith('.IS'):
            ticker += '.IS'
    
    period = st.selectbox(
        "Veri Periyodu",
        ["5y", "10y", "15y", "20y"],
        index=1
    )
    
    st.markdown("---")
    
    st.info("💡 İpucu: 80-80 pivot tespiti kullanılır. Her dip için 100 bar öncesi detaylı analiz yapılır.")

if analysis_mode == "Tek Hisse Analizi":
    st.header(f"🔍 {ticker} Detaylı Dip Analizi")
    
    if st.button("Analizi Başlat", type="primary"):
        with st.spinner(f"{ticker} analiz ediliyor..."):
            result = st.session_state.screener.screen_single_stock(ticker, period)
            
            if result:
                st.success(f"✅ {ticker} için {len(result['dip_analyses'])} dip bulundu!")
                
                dna = result['dna']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Toplam Dip",
                        dna['total_dips'],
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Başarılı Dip",
                        dna['successful_dips'],
                        delta=f"{dna['success_rate']:.1f}%"
                    )
                
                with col3:
                    if dna['all']:
                        st.metric(
                            "Ort. RSI (Dip)",
                            f"{dna['all']['dip_rsi_median']:.1f}",
                            delta=None
                        )
                
                with col4:
                    if dna['all']:
                        st.metric(
                            "Ort. 20G Getiri",
                            f"{dna['all']['avg_ret_20d']:.1f}%",
                            delta=None
                        )
                
                st.markdown("---")
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    " Dip DNA Özeti",
                    " İstatistiksel Analiz",
                    "🎯 Güncel Sinyaller",
                    "📉 Tüm Dipler",
                    "📰 Detaylı Rapor"
                ])
                
                with tab1:
                    st.subheader("🧬 Dip DNA'sı")
                    
                    if dna['all']:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Dip Anı Değerleri")
                            dna_df = pd.DataFrame({
                                'Metrik': [
                                    'RSI (Medyan)',
                                    'MFI (Medyan)',
                                    'StochRSI (Medyan)',
                                    'Williams %R (Medyan)',
                                    'EMA Tangle (Medyan)',
                                    'BB %B (Medyan)',
                                    'Hacim Çarpanı (Medyan)',
                                    'Fiyat Pozisyonu (Medyan)',
                                    'Fibonacci Seviyesi (Medyan)',
                                    'Çekiç Mum Oranı'
                                ],
                                'Değer': [
                                    f"{dna['all']['dip_rsi_median']:.2f}",
                                    f"{dna['all']['dip_mfi_median']:.2f}",
                                    f"{dna['all']['dip_stochrsi_median']:.2f}",
                                    f"{dna['all']['dip_willr_median']:.2f}",
                                    f"{dna['all']['dip_ema_tangle_median']:.2f}%",
                                    f"{dna['all']['dip_bbp_median']:.3f}",
                                    f"{dna['all']['dip_volume_ratio_median']:.2f}x",
                                    f"{dna['all']['dip_price_position_median']:.1f}%",
                                    f"{dna['all']['dip_fib_level_median']:.3f}",
                                    f"{dna['all']['dip_hammer_ratio']*100:.1f}%"
                                ]
                            })
                            st.dataframe(dna_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("### 100 Bar Öncesi İstatistikler")
                            pre_stats_df = pd.DataFrame({
                                'Metrik': [
                                    'Ort. RSI',
                                    'RSI < 30 Gün Sayısı (Ort.)',
                                    'Ort. MFI',
                                    'Ort. EMA Tangle',
                                    'Hacim Patlaması (Ort.)',
                                    'Bearish Gün Sayısı (Ort.)'
                                ],
                                'Değer': [
                                    f"{dna['all']['pre_rsi_mean']:.2f}",
                                    f"{dna['all']['pre_rsi_below_30_avg']:.1f}",
                                    f"{dna['all']['pre_mfi_mean']:.2f}",
                                    f"{dna['all']['pre_ema_tangle_mean']:.2f}%",
                                    f"{dna['all']['pre_volume_spike_avg']:.1f}",
                                    f"{dna['all']['pre_bearish_days_avg']:.1f}"
                                ]
                            })
                            st.dataframe(pre_stats_df, use_container_width=True)
                    
                    if dna['successful'] and dna['failed']:
                        st.markdown("---")
                        st.subheader("✅ Başarılı vs ❌ Başarısız Dip Karşılaştırması")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ✅ Başarılı Dipler (20G +%10)")
                            st.write(f"**Toplam:** {dna['successful']['count']}")
                            st.write(f"**Ort. RSI:** {dna['successful']['dip_rsi_median']:.1f}")
                            st.write(f"**Ort. MFI:** {dna['successful']['dip_mfi_median']:.1f}")
                            st.write(f"**Ort. 20G Getiri:** {dna['successful']['avg_ret_20d']:.1f}%")
                            st.write(f"**Ort. Maks. Kâr:** {dna['successful']['avg_max_profit']:.1f}%")
                        
                        with col2:
                            st.markdown("#### ❌ Başarısız Dipler")
                            st.write(f"**Toplam:** {dna['failed']['count']}")
                            st.write(f"**Ort. RSI:** {dna['failed']['dip_rsi_median']:.1f}")
                            st.write(f"**Ort. MFI:** {dna['failed']['dip_mfi_median']:.1f}")
                            st.write(f"**Ort. 20G Getiri:** {dna['failed']['avg_ret_20d']:.1f}%")
                            st.write(f"**Ort. Maks. Drawdown:** {dna['failed']['avg_max_drawdown']:.1f}%")
                    
                    if dna['significance']:
                        st.markdown("---")
                        st.subheader(" İstatistiksel Anlamlılık Testleri (T-Test)")
                        
                        sig_df = pd.DataFrame([
                            {
                                'Metrik': metric,
                                'Başarılı Ort.': f"{data['succ_mean']:.2f}",
                                'Başarısız Ort.': f"{data['fail_mean']:.2f}",
                                'P-Değeri': f"{data['p_value']:.4f}",
                                'Anlamlı': '✅ EVET' if data['significant'] else '❌ HAYIR'
                            }
                            for metric, data in dna['significance'].items()
                        ])
                        
                        st.dataframe(sig_df, use_container_width=True)
                
                with tab2:
                    st.subheader("📊 Detaylı İstatistiksel Analiz")
                    
                    if dna['all']:
                        st.markdown("### Yüzdelik Dağılımlar (Percentiles)")
                        
                        percentiles_df = pd.DataFrame({
                            'Metrik': [
                                'RSI',
                                'MFI',
                                'EMA Tangle',
                                'Hacim Çarpanı',
                                'Fiyat Pozisyonu'
                            ],
                            '25. Yüzdelik': [
                                f"{dna['all']['rsi_14_25pct']:.2f}" if 'rsi_14_25pct' in dna['all'] else "N/A",
                                f"{dna['all']['mfi_14_25pct']:.2f}" if 'mfi_14_25pct' in dna['all'] else "N/A",
                                f"{dna['all']['ema_tangle_25pct']:.2f}" if 'ema_tangle_25pct' in dna['all'] else "N/A",
                                f"{dna['all']['volume_ratio_20_25pct']:.2f}" if 'volume_ratio_20_25pct' in dna['all'] else "N/A",
                                f"{dna['all']['price_position_25pct']:.1f}%" if 'price_position_25pct' in dna['all'] else "N/A"
                            ],
                            'Medyan (50.)': [
                                f"{dna['all']['dip_rsi_median']:.2f}",
                                f"{dna['all']['dip_mfi_median']:.2f}",
                                f"{dna['all']['dip_ema_tangle_median']:.2f}",
                                f"{dna['all']['dip_volume_ratio_median']:.2f}",
                                f"{dna['all']['dip_price_position_median']:.1f}%"
                            ],
                            '75. Yüzdelik': [
                                f"{dna['all']['rsi_14_75pct']:.2f}" if 'rsi_14_75pct' in dna['all'] else "N/A",
                                f"{dna['all']['mfi_14_75pct']:.2f}" if 'mfi_14_75pct' in dna['all'] else "N/A",
                                f"{dna['all']['ema_tangle_75pct']:.2f}" if 'ema_tangle_75pct' in dna['all'] else "N/A",
                                f"{dna['all']['volume_ratio_20_75pct']:.2f}" if 'volume_ratio_20_75pct' in dna['all'] else "N/A",
                                f"{dna['all']['price_position_75pct']:.1f}%" if 'price_position_75pct' in dna['all'] else "N/A"
                            ]
                        })
                        
                        st.dataframe(percentiles_df, use_container_width=True)
                
                with tab3:
                    st.subheader("🎯 Güncel Sinyaller (Son 30 Gün)")
                    
                    with st.spinner("Güncel sinyaller taranıyor..."):
                        signals = st.session_state.screener.find_current_signals(ticker, dna, lookback_days=30)
                    
                    if signals:
                        st.success(f"✅ {len(signals)} sinyal bulundu!")
                        
                        signals_df = pd.DataFrame(signals)
                        signals_df = signals_df.sort_values('dna_match_score', ascending=False)
                        
                        def get_status(row):
                            if row['price_increase_pct'] < -5:
                                return "⚠️ Zararda"
                            elif row['price_increase_pct'] < 3:
                                return "🟡 Bekleme"
                            elif row['price_increase_pct'] < 8:
                                return "🟢 Hareket Başladı"
                            else:
                                return "🔴 Fırsat Kaçmış"
                        
                        signals_df['Durum'] = signals_df.apply(get_status, axis=1)
                        
                        st.dataframe(
                            signals_df[['date', 'price', 'current_price', 'price_increase_pct', 
                                       'dna_match_score', 'rsi', 'mfi', 'Durum']],
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ Son 30 günde DNA'ya uyan sinyal bulunamadı.")
                
                with tab4:
                    st.subheader("📉 Tüm Bulunan Dipler")
                    
                    dips_data = []
                    for i, analysis in enumerate(result['dip_analyses']):
                        dip_bar = analysis['dip_bar']
                        post_stats = analysis['post_stats']
                        
                        dips_data.append({
                            'Tarih': dip_bar['date'],
                            'Fiyat': f"{dip_bar['close']:.2f}",
                            'RSI': f"{dip_bar['rsi_14']:.1f}",
                            'MFI': f"{dip_bar['mfi_14']:.1f}",
                            'EMA Tangle': f"{dip_bar['ema_tangle']:.2f}%",
                            '5G Getiri': f"{post_stats.get('ret_5d', 0):.1f}%",
                            '20G Getiri': f"{post_stats.get('ret_20d', 0):.1f}%",
                            'Maks. Kâr': f"{post_stats.get('max_profit', 0):.1f}%",
                            'Maks. Drawdown': f"{post_stats.get('max_drawdown', 0):.1f}%",
                            'Başarılı': '✅' if post_stats.get('ret_20d', 0) >= 10 else '❌'
                        })
                    
                    dips_df = pd.DataFrame(dips_data)
                    st.dataframe(dips_df, use_container_width=True)
                
                with tab5:
                    st.subheader(" Detaylı Analiz Raporu")
                    
                    df = result['df']
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Fiyat'
                    ))
                    
                    pivot_dates = [df.index[i] for i in result['pivot_indices']]
                    pivot_prices = [df['Low'].iloc[i] for i in result['pivot_indices']]
                    
                    fig.add_trace(go.Scatter(
                        x=pivot_dates,
                        y=pivot_prices,
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red',
                            line=dict(color='DarkSlateGray', width=2)
                        ),
                        name='Pivot Dipler'
                    ))
                    
                    for period in [20, 50, 200]:
                        col = f'EMA_{period}'
                        if col in df.columns:
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df[col],
                                mode='lines',
                                name=f'EMA {period}',
                                line=dict(width=1)
                            ))
                    
                    fig.update_layout(
                        title=f"{ticker} Fiyat ve Pivot Dipler",
                        yaxis_title='Fiyat',
                        xaxis_title='Tarih',
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### İndikatörler")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'RSI_14' in df.columns:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(
                                x=df.index,
                                y=df['RSI_14'],
                                mode='lines',
                                name='RSI 14'
                            ))
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                            fig_rsi.update_layout(title='RSI', height=300, yaxis_range=[0, 100])
                            st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    with col2:
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Bar(
                            x=df.index,
                            y=df['Volume'],
                            name='Hacim'
                        ))
                        fig_vol.update_layout(title='Hacim', height=300)
                        st.plotly_chart(fig_vol, use_container_width=True)
                
            else:
                st.error(f"❌ {ticker} için yeterli veri bulunamadı veya dip tespit edilemedi.")
    
    else:
        st.info("👆 Analizi başlatmak için yukarıdaki butona tıklayın.")

elif analysis_mode == "Çoklu Hisse Tarama":
    st.header("🔍 Çoklu Hisse Tarama")
    
    tickers_input = st.text_area(
        "Hisse Kodları (her satıra bir hisse)",
        "THYAO.IS\nASELS.IS\nGARAN.IS\nEREGL.IS\nSISE.IS",
        height=150
    )
    
    tickers = [t.strip().upper() for t in tickers_input.split('\n') if t.s
