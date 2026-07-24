import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from tradingview_screener import get_all_symbols
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================================
# BÖLÜM 1: YARDIMCI FONKSİYONLAR
# ============================================================================

@st.cache_data(ttl=3600)
def get_all_bist_tickers():
    try:
        symbols = get_all_symbols(market='turkey')
        return [s.split(':')[1] + '.IS' for s in symbols if 'BIST:' in s]
    except:
        return ["THYAO.IS", "ASELS.IS", "GARAN.IS", "EREGL.IS"]


def find_80_80_pivots(df):
    lows = df['Low'].values
    n = len(lows)
    pivots = []
    for i in range(80, n - 80):
        if np.all(lows[i] < lows[i-80:i]) and np.all(lows[i] < lows[i+1:i+81]):
            pivots.append(i)
    return pivots


def is_business_day(date_str):
    return pd.to_datetime(date_str).weekday() < 5


# ============================================================================
# BÖLÜM 2: İNDİKATÖR HESAPLAMA (pandas-ta OLMADAN)
# ============================================================================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos = mf.where(delta > 0, 0).rolling(window=period).sum()
    neg = mf.where(delta < 0, 0).rolling(window=period).sum()
    return 100 - (100 / (1 + pos / neg))


def calc_stoch_rsi(rsi, period=14, k=3, d=3):
    low_rsi = rsi.rolling(window=period).min()
    high_rsi = rsi.rolling(window=period).max()
    stoch = 100 * ((rsi - low_rsi) / (high_rsi - low_rsi))
    k_line = stoch.rolling(window=k).mean()
    d_line = k_line.rolling(window=d).mean()
    return k_line, d_line


def calc_williams_r(high, low, close, period=14):
    hh = high.rolling(window=period).max()
    ll = low.rolling(window=period).min()
    return -100 * ((hh - close) / (hh - ll))


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_bbands(close, period=20, std_dev=2):
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + (std_dev * std)
    lower = mid - (std_dev * std)
    pct = (close - lower) / (upper - lower)
    width = ((upper - lower) / mid) * 100
    return upper, mid, lower, pct, width


def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def add_indicators(df):
    if df is None or len(df) < 100:
        return None
    df = df.copy()
    for p in [8, 13, 21, 50, 89, 200]:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    for p in [10, 20, 50, 100, 200]:
        df[f'SMA_{p}'] = df['Close'].rolling(window=p).mean()
    df['RSI_14'] = calc_rsi(df['Close'], 14)
    df['MFI_14'] = calc_mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)
    k, d = calc_stoch_rsi(df['RSI_14'], 14, 3, 3)
    df['STOCHRSIk_14_14_3_3'] = k
    df['STOCHRSId_14_14_3_3'] = d
    df['WILLR_14'] = calc_williams_r(df['High'], df['Low'], df['Close'], 14)
    m, s, h = calc_macd(df['Close'], 12, 26, 9)
    df['MACD_12_26_9'] = m
    df['MACDs_12_26_9'] = s
    df['MACDh_12_26_9'] = h
    u, mid, l, p, w = calc_bbands(df['Close'], 20, 2)
    df['BBU_20_2.0'] = u
    df['BBM_20_2.0'] = mid
    df['BBL_20_2.0'] = l
    df['BBP_20_2.0'] = p
    df['BBB_20_2.0'] = w
    df['ATR_14'] = calc_atr(df['High'], df['Low'], df['Close'], 14)
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    return df


# ============================================================================
# BÖLÜM 3: VERİ ÇEKME VE DİP ANALİZİ
# ============================================================================

def download_stock(ticker, period="10y"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, timeout=15)
        if df.empty or len(df) < 300:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if isinstance(df['Volume'], pd.DataFrame):
            df['Volume'] = df['Volume'].iloc[:, 0]
        for col in ['Open', 'High', 'Low', 'Close']:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
        df = df.ffill().dropna()
        if len(df) < 300:
            return None
        return df
    except:
        return None


def analyze_dip(df, dip_idx):
    if dip_idx < 100 or dip_idx >= len(df) - 60:
        return None
    row = df.iloc[dip_idx]
    date_str = df.index[dip_idx].strftime('%Y-%m-%d')
    if not is_business_day(date_str):
        return None
    close_p = float(row['Close'])
    rsi = float(row.get('RSI_14', 50))
    mfi = float(row.get('MFI_14', 50))
    stoch = float(row.get('STOCHRSIk_14_14_3_3', 50))
    willr = float(row.get('WILLR_14', -50))
    ma_vals = []
    for p in [8, 13, 21, 50, 89, 200]:
        col = f'EMA_{p}'
        if col in df.columns:
            val = float(row.get(col, np.nan))
            if pd.notna(val) and val > 0:
                ma_vals.append(val)
    ma_tangle = (np.std(ma_vals) / np.mean(ma_vals)) * 100 if len(ma_vals) > 1 and np.mean(ma_vals) > 0 else 50
    vol_sma = float(row.get('Volume_SMA_20', 0))
    vol_ratio = float(row['Volume']) / vol_sma if vol_sma > 0 else 1.0
    lowest_20 = df['Low'].iloc[max(0, dip_idx-19):dip_idx+1].min()
    highest_20 = df['High'].iloc[max(0, dip_idx-19):dip_idx+1].max()
    range_20 = highest_20 - lowest_20
    price_pos = ((close_p - lowest_20) / range_20) * 100 if range_20 > 0 else 50
    if dip_idx + 20 < len(df):
        ret_20d = ((float(df.iloc[dip_idx + 20]['Close']) / close_p) - 1) * 100
    else:
        ret_20d = np.nan
    return {
        'date': date_str,
        'price': close_p,
        'rsi': rsi,
        'mfi': mfi,
        'stoch': stoch,
        'willr': willr,
        'ma_tangle': ma_tangle,
        'vol_ratio': vol_ratio,
        'price_pos': price_pos,
        'ret_20d': ret_20d,
        'is_successful': 1 if pd.notna(ret_20d) and ret_20d >= 10.0 else 0
    }


def process_stock(ticker, period="10y"):
    df = download_stock(ticker, period)
    if df is None:
        return None, None
    df = add_indicators(df)
    pivots = find_80_80_pivots(df)
    if len(pivots) < 1:
        return df, None
    dips = []
    for p_idx in pivots:
        res = analyze_dip(df, p_idx)
        if res:
            res['ticker'] = ticker
            dips.append(res)
    if not dips:
        return df, None
    return df, pd.DataFrame(dips)


# ============================================================================
# BÖLÜM 4: CANLI TARAMA
# ============================================================================

def find_live_signals(ticker, lookback_days=30):
    df = download_stock(ticker, period="6mo")
    if df is None:
        return []
    df = add_indicators(df)
    if df is None or len(df) < lookback_days:
        return []
    current_price = float(df.iloc[-1]['Close'])
    signals = []
    for i in range(len(df)-lookback_days, len(df)):
        row = df.iloc[i]
        date = df.index[i].strftime('%Y-%m-%d')
        if not is_business_day(date):
            continue
        close = float(row['Close'])
        price_increase = ((current_price / close) - 1) * 100
        if price_increase >= 10.0:
            continue
        rsi = float(row.get('RSI_14', 50))
        mfi = float(row.get('MFI_14', 50))
        stoch = float(row.get('STOCHRSIk_14_14_3_3', 50))
        willr = float(row.get('WILLR_14', -50))
        ma_vals = []
        for p in [8, 13, 21, 50, 89, 200]:
            col = f'EMA_{p}'
            if col in df.columns:
                val = float(row.get(col, np.nan))
                if pd.notna(val) and val > 0:
                    ma_vals.append(val)
        ma_tangle = (np.std(ma_vals) / np.mean(ma_vals)) * 100 if len(ma_vals) > 1 and np.mean(ma_vals) > 0 else 50
        bb_upper = float(row.get('BBU_20_2.0', np.nan))
        bb_lower = float(row.get('BBL_20_2.0', np.nan))
        bb_pct = (close - bb_lower) / (bb_upper - bb_lower) if (bb_upper > bb_lower) else 0.5
        vol_sma = float(row.get('Volume_SMA_20', 0))
        vol_ratio = float(row['Volume']) / vol_sma if vol_sma > 0 else 1.0
        lowest_20 = df['Low'].iloc[max(0, i-19):i+1].min()
        highest_20 = df['High'].iloc[max(0, i-19):i+1].max()
        range_20 = highest_20 - lowest_20
        price_pos = ((close - lowest_20) / range_20) * 100 if range_20 > 0 else 50
        if rsi < 45 and stoch < 35 and price_pos < 30:
            signals.append({
                'date': date,
                'price': close,
                'current_price': current_price,
                'price_increase_pct': round(price_increase, 2),
                'rsi': round(rsi, 1),
                'mfi': round(mfi, 1),
                'stoch': round(stoch, 1),
                'willr': round(willr, 1),
                'ma_tangle': round(ma_tangle, 2),
                'bb_pct': round(bb_pct, 3),
                'vol_ratio': round(vol_ratio, 2),
                'price_pos': round(price_pos, 1)
            })
    return signals


# ============================================================================
# BÖLÜM 5: STREAMLIT ARAYÜZÜ
# ============================================================================

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚀 ULTRA PRO DİP ANALİZ SİSTEMİ</div>', unsafe_allow_html=True)
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
        ticker_input = st.text_input("Hisse Kodu", "THYAO.IS").upper().strip()
        if not ticker_input.endswith('.IS'):
            ticker_input += '.IS'
    elif analysis_mode == "Çoklu Hisse Tarama":
        tickers_text = st.text_area(
            "Hisse Kodları (her satıra bir hisse)",
            "THYAO.IS\nASELS.IS\nGARAN.IS\nEREGL.IS\nSISE.IS",
            height=150
        )
    period = st.selectbox("Veri Periyodu", ["5y", "10y", "15y", "20y"], index=1)
    st.markdown("---")
    st.info(" 80-80 pivot tespiti kullanılır. Her dip için detaylı analiz yapılır.")

# ============================================================================
# MOD 1: TEK HİSSE ANALİZİ
# ============================================================================

if analysis_mode == "Tek Hisse Analizi":
    st.header(f"🔍 {ticker_input} Detaylı Dip Analizi")
    if st.button("Analizi Başlat", type="primary"):
        with st.spinner(f"{ticker_input} analiz ediliyor..."):
            df, dips_df = process_stock(ticker_input, period)
        if dips_df is not None and not dips_df.empty:
            st.success(f"✅ {ticker_input} için {len(dips_df)} adet 80-80 pivot dibi bulundu!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Toplam Dip", len(dips_df))
            col2.metric("Başarılı Dip (%10+)", f"{dips_df['is_successful'].sum()} ({dips_df['is_successful'].mean()*100:.1f}%)")
            col3.metric("Ort. RSI", f"{dips_df['rsi'].mean():.1f}")
            col4.metric("Ort. 20G Getiri", f"{dips_df['ret_20d'].mean():.1f}%")
            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["📊 Tüm Dipler", "🎯 Güncel Sinyaller", " Grafik"])
            with tab1:
                st.subheader("Bulunan Dipler ve Sonrası Performans")
                display_df = dips_df[['date', 'price', 'rsi', 'mfi', 'stoch', 'willr', 'ma_tangle', 'vol_ratio', 'price_pos', 'ret_20d', 'is_successful']].copy()
                display_df.columns = ['Tarih', 'Fiyat', 'RSI', 'MFI', 'StochRSI', 'Williams_R', 'MA_Tangle', 'Hacim_Çarpanı', 'Fiyat_Pozisyonu', '20G_Getiri', 'Başarılı']
                display_df['Başarılı'] = display_df['Başarılı'].apply(lambda x: '✅' if x == 1 else '❌')
                st.dataframe(display_df.sort_values('Tarih', ascending=False), use_container_width=True)
                csv = dips_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
                st.download_button("📥 CSV İndir", data=csv, file_name=f"{ticker_input}_dip_analiz.csv", mime="text/csv")
            with tab2:
                st.subheader("Son 30 Günde Sinyal")
                with st.spinner("Taranıyor..."):
                    live_signals = find_live_signals(ticker_input, lookback_days=30)
                if live_signals:
                    st.success(f"✅ {len(live_signals)} sinyal bulundu!")
                    sig_df = pd.DataFrame(live_signals)
                    sig_df = sig_df.sort_values('rsi', ascending=True)
                    st.dataframe(sig_df, use_container_width=True)
                else:
                    st.warning("️ Son 30 günde kriterlere uyan sinyal yok.")
            with tab3:
                st.subheader(f"{ticker_input} Fiyat ve Pivot Dipler")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Fiyat'
                ))
                pivot_indices = find_80_80_pivots(df)
                pivot_dates = [df.index[i] for i in pivot_indices]
                pivot_prices = [df['Low'].iloc[i] for i in pivot_indices]
                fig.add_trace(go.Scatter(
                    x=pivot_dates, y=pivot_prices, mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'), name='80-80 Pivot Dipler'
                ))
                for p in [20, 50, 200]:
                    col = f'EMA_{p}'
                    if col in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df[col], mode='lines', name=f'EMA {p}', line=dict(width=1)
                        ))
                fig.update_layout(title=f"{ticker_input} Fiyat ve Pivot Dipler", height=500, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"⚠️ {ticker_input} için yeterli veri veya 80-80 pivot dibi bulunamadı.")

# ============================================================================
# MOD 2: ÇOKLU HİSSE TARAMA
# ============================================================================

elif analysis_mode == "Çoklu Hisse Tarama":
    st.header(" Çoklu Hisse Tarama")
    tickers_list = [t.strip().upper() for t in tickers_text.split('\n') if t.strip()]
    tickers_list = [t if t.endswith('.IS') else t + '.IS' for t in tickers_list]
    if st.button("Taramayı Başlat", type="primary"):
        with st.spinner(f"{len(tickers_list)} hisse taranıyor..."):
            progress_bar = st.progress(0)
            all_results = []
            for i, t in enumerate(tickers_list):
                df, dips_df = process_stock(t, period)
                if dips_df is not None and not dips_df.empty:
                    all_results.append(dips_df)
                progress_bar.progress((i + 1) / len(tickers_list))
        if all_results:
            master_df = pd.concat(all_results, ignore_index=True)
            st.success(f"✅ {len(master_df)} toplam dip bulundu!")
            summary = master_df.groupby('ticker').agg(
                Dip_Sayisi=('ticker', 'count'),
                Basarili_Dip=('is_successful', 'sum'),
                Basari_Orani=('is_successful', lambda x: f"{x.mean()*100:.1f}%"),
                Ort_RSI=('rsi', 'mean'),
                Ort_20G_Getiri=('ret_20d', 'mean')
            ).sort_values('Basarili_Dip', ascending=False).reset_index()
            st.dataframe(summary, use_container_width=True)
            csv = master_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
            st.download_button("📥 Tüm Veriyi İndir (CSV)", data=csv, file_name=f"tum_dip_analizleri_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
        else:
            st.warning("⚠️ Hiçbir hisse için dip bulunamadı.")

# ============================================================================
# MOD 3: TÜM BIST TARAMA
# ============================================================================

elif analysis_mode == "Tüm BIST Tarama":
    st.header("🌟 Tüm BIST Tarama")
    st.warning("⚠️ Bu işlem 600+ hisse için çalışır ve 10-15 dakika sürebilir.")
    if st.button("Tüm BIST'i Tara", type="primary"):
        all_tickers = get_all_bist_tickers()
        st.write(f"Toplam {len(all_tickers)} hisse taranacak...")
        progress_bar = st.progress(0)
        all_results = []
        start_time = time.time()
        for i, t in enumerate(all_tickers):
            try:
                df, dips_df = process_stock(t, period)
                if dips_df is not None and not dips_df.empty:
                    all_results.append(dips_df)
            except:
                pass
            if i % 10 == 0:
                progress_bar.progress((i + 1) / len(all_tickers))
        progress_bar.progress(1.0)
        elapsed = time.time() - start_time
        st.success(f"✅ Tarama tamamlandı! Süre: {elapsed:.0f} saniye. {len(all_results)} hisse için dip bulundu.")
        if all_results:
            master_df = pd.concat(all_results, ignore_index=True)
            st.subheader("🏆 En Çok Başarılı Dip Üreten Hisseler")
            summary = master_df.groupby('ticker').agg(
                Dip_Sayisi=('ticker', 'count'),
                Basarili_Dip=('is_successful', 'sum'),
                Basari_Orani=('is_successful', lambda x: f"{x.mean()*100:.1f}%"),
                Ort_RSI=('rsi', 'mean'),
                Ort_20G_Getiri=('ret_20d', 'mean')
            ).sort_values('Basarili_Dip', ascending=False).reset_index()
            st.dataframe(summary.head(50), use_container_width=True)
            csv = master_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
            st.download_button("📥 Tüm BIST Verisini İndir (CSV)", data=csv, file_name=f"bist_tum_dipler_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'><p>Ultra Pro Dip Analiz Sistemi v2.0 | 80-80 Pivot + Manuel İndikatörler</p></div>", unsafe_allow_html=True)
