"""
==============================================================
  ARIA SYAF XGBoost Multi-Asset Trainer v3.0
  تدريب نماذج منفصلة لكل فئة أصول تلقائياً:
    - BTC/ETH (كريبتو)
    - XAUUSD (ذهب)
    - Forex (EURUSD, GBPUSD, USDJPY, ...)
==============================================================
الاستخدام:
  python train_multi_model.py --btc  BTCUSD_M1.csv
                               --gold XAUUSD_M1.csv
                               --forex EURUSD_M1.csv GBPUSD_M1.csv
  سيُنشئ: models/btc_model.pkl  models/gold_model.pkl  models/forex_model.pkl
==============================================================
"""

import argparse
import os
import math
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings('ignore')
os.makedirs("models", exist_ok=True)

# ================================================================
#  إعدادات كل فئة أصول
# ================================================================
ASSET_CONFIG = {
    "btc": {
        "name"           : "BTC/Crypto",
        "fast_ema"       : 10,
        "slow_ema"       : 50,
        "adx_period"     : 14,
        "adx_threshold"  : 28,    # BTC يحتاج ADX أعلى
        "atr_period"     : 14,
        "rsi_period"     : 14,
        "bb_period"      : 20,
        "future_bars"    : 5,
        "min_profit_atr" : 1.5,
        "pip_multiplier" : 1.0,   # BTC: النقطة = 1 دولار
        "description"    : "Bitcoin & Crypto — تقلب عالٍ، زخم قوي",
    },
    "gold": {
        "name"           : "XAUUSD/Gold",
        "fast_ema"       : 8,
        "slow_ema"       : 21,
        "adx_period"     : 14,
        "adx_threshold"  : 25,
        "atr_period"     : 14,
        "rsi_period"     : 14,
        "bb_period"      : 20,
        "future_bars"    : 5,
        "min_profit_atr" : 1.2,
        "pip_multiplier" : 10.0,  # Gold: النقطة = 0.1 دولار
        "description"    : "XAUUSD — تقلب متوسط، حساس للأخبار",
    },
    "forex": {
        "name"           : "Forex Majors",
        "fast_ema"       : 10,
        "slow_ema"       : 50,
        "adx_period"     : 14,
        "adx_threshold"  : 22,
        "atr_period"     : 14,
        "rsi_period"     : 14,
        "bb_period"      : 20,
        "future_bars"    : 5,
        "min_profit_atr" : 1.0,
        "pip_multiplier" : 10000.0,  # Forex: النقطة = 0.0001
        "description"    : "EUR/USD, GBP/USD, USD/JPY — تقلب منخفض، اتجاهات واضحة",
    },
}

# ================================================================
#  دوال المؤشرات التقنية
# ================================================================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()

def calc_rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = ema(m, signal)
    return m, s, m - s

def calc_stoch(high, low, close, k=5, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    sk = 100 * (close - lo) / (hi - lo + 1e-10)
    return sk, sk.rolling(d).mean()

def calc_bb(close, period=20, std=2):
    mid = sma(close, period)
    s   = close.rolling(period).std()
    up  = mid + std * s
    dn  = mid - std * s
    w   = up - dn
    pos = (close - dn) / (w + 1e-10)
    return up, mid, dn, w, pos

def calc_adx(high, low, close, period=14):
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    up   = high - high.shift()
    dn   = low.shift() - low
    pdm  = np.where((up > dn) & (up > 0), up, 0.0)
    mdm  = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_ = pd.Series(tr).rolling(period).mean()
    pdi  = 100 * pd.Series(pdm).rolling(period).mean() / (atr_ + 1e-10)
    mdi  = 100 * pd.Series(mdm).rolling(period).mean() / (atr_ + 1e-10)
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)
    return dx.rolling(period).mean(), pdi, mdi

# ================================================================
#  بناء الميزات (مشتركة + مخصصة لكل فئة)
# ================================================================
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c, h, l, o = df['close'], df['high'], df['low'], df['open']

    # ── EMA ──
    df['fast_ema']  = ema(c, cfg['fast_ema'])
    df['slow_ema']  = ema(c, cfg['slow_ema'])
    df['ema200']    = ema(c, 200)
    df['ema_cross'] = np.sign(df['fast_ema'] - df['slow_ema'])
    df['price_vs_ema200'] = (c - df['ema200']) / (df['ema200'] + 1e-10)
    df['price_vs_fast']   = (c - df['fast_ema']) / (df['fast_ema'] + 1e-10)

    # ── ADX ──
    adx, pdi, mdi = calc_adx(h, l, c, cfg['adx_period'])
    df['adx'], df['di_plus'], df['di_minus'] = adx, pdi, mdi
    df['adx_above_thresh'] = (adx > cfg['adx_threshold']).astype(float)

    # ── ATR متعدد ──
    df['atr14']     = calc_atr(h, l, c, 14)
    df['atr7']      = calc_atr(h, l, c, 7)
    df['atr21']     = calc_atr(h, l, c, 21)
    df['atr_ratio'] = df['atr7'] / (df['atr21'] + 1e-10)

    # ── RSI ──
    df['rsi14'] = calc_rsi(c, 14)
    df['rsi7']  = calc_rsi(c, 7)
    df['rsi_overbought'] = (df['rsi14'] > 70).astype(float)
    df['rsi_oversold']   = (df['rsi14'] < 30).astype(float)

    # ── MACD ──
    df['macd_main'], df['macd_signal'], df['macd_hist'] = calc_macd(c)
    df['macd_cross'] = np.sign(df['macd_main'] - df['macd_signal'])

    # ── Stochastic ──
    df['stoch_k'], df['stoch_d'] = calc_stoch(h, l, c)
    df['stoch_cross'] = np.sign(df['stoch_k'] - df['stoch_d'])

    # ── Bollinger ──
    _, _, _, df['bb_width'], df['bb_position'] = calc_bb(c, cfg['bb_period'])
    df['bb_width_norm'] = df['bb_width'] / (c + 1e-10)
    df['bb_squeeze']    = (df['bb_width_norm'] < df['bb_width_norm'].rolling(50).quantile(0.2)).astype(float)

    # ── هيكل الشمعات ──
    atr_s = df['atr14'].replace(0, 1)
    df['candle_body']        = (c - o).abs() / atr_s
    df['candle_upper_wick']  = (h - pd.concat([c,o], axis=1).max(axis=1)) / atr_s
    df['candle_lower_wick']  = (pd.concat([c,o], axis=1).min(axis=1) - l) / atr_s
    df['candle_direction']   = np.sign(c - o)
    df['close_change1']      = (c - c.shift(1)) / atr_s
    df['close_change2']      = (c - c.shift(2)) / atr_s
    df['close_change3']      = (c - c.shift(3)) / atr_s
    df['trend_3bars']        = (np.sign(c-c.shift(1)) + np.sign(c.shift(1)-c.shift(2)) + np.sign(c.shift(2)-c.shift(3))) / 3.0

    # ── High/Low نسبي ──
    df['high20'] = h.rolling(20).max()
    df['low20']  = l.rolling(20).min()
    df['price_in_range'] = (c - df['low20']) / (df['high20'] - df['low20'] + 1e-10)

    # ── السياق الزمني ──
    df['hour']    = df['time'].dt.hour
    df['dow']     = df['time'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['dow'] / 7)

    # ── ميزة خاصة: فئة الأصل (0=forex, 1=gold, 2=btc) ──
    asset_map = {"forex": 0.0, "gold": 0.5, "btc": 1.0}
    df['asset_class'] = asset_map.get(cfg.get('_key', 'forex'), 0.0)

    return df

# ================================================================
#  قائمة الميزات الرسمية (44 ميزة)
# ================================================================
FEATURE_COLUMNS = [
    'fast_ema','slow_ema','ema200','ema_cross','price_vs_ema200','price_vs_fast',
    'adx','di_plus','di_minus','adx_above_thresh',
    'atr14','atr7','atr21','atr_ratio',
    'rsi14','rsi7','rsi_overbought','rsi_oversold',
    'macd_main','macd_signal','macd_hist','macd_cross',
    'stoch_k','stoch_d','stoch_cross',
    'bb_width','bb_width_norm','bb_position','bb_squeeze',
    'candle_body','candle_upper_wick','candle_lower_wick','candle_direction',
    'close_change1','close_change2','close_change3','trend_3bars',
    'price_in_range',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'asset_class',
]

# ================================================================
#  بناء الهدف
# ================================================================
def build_target(df, future_bars=5, min_profit_atr=1.5):
    c   = df['close']
    atr = df['atr14']
    fh  = c.rolling(future_bars).max().shift(-future_bars)
    fl  = c.rolling(future_bars).min().shift(-future_bars)
    thr = min_profit_atr * atr
    df['target'] = 0
    df.loc[(fh - c) > thr, 'target'] = 1
    df.loc[(c - fl) > thr, 'target'] = -1
    return df

# ================================================================
#  تدريب نموذج واحد
# ================================================================
def train_one(df, cfg, asset_key):
    cfg['_key'] = asset_key
    df = build_features(df, cfg)
    df = build_target(df, cfg['future_bars'], cfg['min_profit_atr'])
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target']).copy()

    print(f"\n{'='*55}")
    print(f"  تدريب نموذج: {cfg['name']} ({len(df_clean):,} شمعة)")
    print(f"{'='*55}")

    X = df_clean[FEATURE_COLUMNS].values
    y = df_clean['target'].values
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # نموذج الاتجاه
    model_dir = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1, early_stopping_rounds=30,
    )
    model_dir.fit(X_tr, y_tr+1, eval_set=[(X_te, y_te+1)], verbose=50)

    y_pred = model_dir.predict(X_te) - 1
    print(classification_report(y_te, y_pred, target_names=['SELL','NEUTRAL','BUY']))

    # نموذج SL
    y_sl = df_clean['atr14'].values / (df_clean['close'].values + 1e-10) * cfg['pip_multiplier'] * 10000
    model_sl = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model_sl.fit(X_tr, y_sl[:split])

    # نموذج TP
    y_tp = y_sl * 2.5
    model_tp = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    model_tp.fit(X_tr, y_tp[:split])

    # أهمية الميزات
    imp = pd.Series(model_dir.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print(f"\n🔍 أهم 10 ميزات لـ {cfg['name']}:")
    for i, (f, s) in enumerate(imp.head(10).items()):
        print(f"  {i+1:2}. {f:<28} {'█'*int(s*200)} {s:.4f}")

    return {
        'model_dir'      : model_dir,
        'model_sl'       : model_sl,
        'model_tp'       : model_tp,
        'feature_columns': FEATURE_COLUMNS,
        'config'         : cfg,
        'version'        : '3.0',
        'asset_key'      : asset_key,
    }

# ================================================================
#  التشغيل الرئيسي
# ================================================================
def load_csv(path):
    # 1. قراءة الملف بشكل طبيعي
    df = pd.read_csv(path)
    
    # 2. تنظيف أسماء الأعمدة (تحويل لحروف صغيرة وإزالة أقواس الميتاتريدر < >)
    df.columns = df.columns.str.lower().str.replace('<', '').str.replace('>', '')
    
    # 3. معالجة التاريخ والوقت بذكاء
    if 'date' in df.columns and 'time' in df.columns:
        # دمج عمود التاريخ مع عمود الوقت (صيغة الميتاتريدر المعتادة)
        df['time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
        df.drop(columns=['date'], inplace=True)
    elif 'date' in df.columns and 'time' not in df.columns:
        # إذا كان هناك عمود تاريخ فقط
        df.rename(columns={'date': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
    else:
        # إذا كان العمود اسمه time وجاهزاً
        df['time'] = pd.to_datetime(df['time'])

    # 4. الترتيب وإعادة الضبط (هذا الجزء الذي كان ناقصاً)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  ✅ {path}: {len(df):,} شمعة")
    
    # 5. إرجاع البيانات للكود الرئيسي (الأهم!)
    return df
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARIA Multi-Asset XGBoost Trainer v3.0')
    parser.add_argument('--btc',   nargs='+', help='ملفات CSV للكريبتو (BTC, ETH)')
    parser.add_argument('--gold',  nargs='+', help='ملفات CSV للذهب (XAUUSD)')
    parser.add_argument('--forex', nargs='+', help='ملفات CSV للفوركس (EURUSD, GBPUSD...)')
    args = parser.parse_args()

    print("=" * 55)
    print("  ARIA SYAF Multi-Asset XGBoost Trainer v3.0")
    print("=" * 55)

    tasks = [
        ('btc',   args.btc,   ASSET_CONFIG['btc']),
        ('gold',  args.gold,  ASSET_CONFIG['gold']),
        ('forex', args.forex, ASSET_CONFIG['forex']),
    ]

    for key, files, cfg in tasks:
        if not files:
            print(f"\n⚠️  تخطي {key} — لم يُحدَّد ملف بيانات")
            continue
        # دمج ملفات متعددة (مثلاً EURUSD + GBPUSD معاً للفوركس)
        frames = [load_csv(f) for f in files]
        df = pd.concat(frames, ignore_index=True).sort_values('time').reset_index(drop=True)
        pkg = train_one(df, cfg.copy(), key)
        out = f"models/{key}_model.pkl"
        joblib.dump(pkg, out)
        print(f"\n✅ تم حفظ نموذج {cfg['name']} في: {out}")

    print("\n🎉 اكتمل التدريب! شغّل الآن: python app.py")
