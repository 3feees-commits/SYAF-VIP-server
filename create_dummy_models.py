"""
==============================================================
  إنشاء نماذج وهمية مؤقتة للاختبار
  شغّل هذا السكريبت مرة واحدة قبل رفع الملفات على GitHub
  لاحقاً استبدلها بالنماذج الحقيقية بعد التدريب
==============================================================
الاستخدام:
  python create_dummy_models.py
==============================================================
"""

import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

os.makedirs("models", exist_ok=True)

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

ASSET_CONFIG = {
    "btc": {
        "name": "BTC/Crypto", "fast_ema": 10, "slow_ema": 50,
        "adx_period": 14, "adx_threshold": 28, "atr_period": 14,
        "rsi_period": 14, "bb_period": 20, "future_bars": 5,
        "min_profit_atr": 1.5, "pip_multiplier": 1.0,
        "description": "Bitcoin & Crypto — تقلب عالٍ",
    },
    "gold": {
        "name": "XAUUSD/Gold", "fast_ema": 8, "slow_ema": 21,
        "adx_period": 14, "adx_threshold": 25, "atr_period": 14,
        "rsi_period": 14, "bb_period": 20, "future_bars": 5,
        "min_profit_atr": 1.2, "pip_multiplier": 10.0,
        "description": "XAUUSD — تقلب متوسط",
    },
    "forex": {
        "name": "Forex Majors", "fast_ema": 10, "slow_ema": 50,
        "adx_period": 14, "adx_threshold": 22, "atr_period": 14,
        "rsi_period": 14, "bb_period": 20, "future_bars": 5,
        "min_profit_atr": 1.0, "pip_multiplier": 10000.0,
        "description": "EUR/USD, GBP/USD — تقلب منخفض",
    },
}

n_features = len(FEATURE_COLUMNS)

for key, cfg in ASSET_CONFIG.items():
    print(f"⚙️  إنشاء نموذج وهمي لـ {cfg['name']}...")

    # بيانات عشوائية للتدريب السريع
    X, y_cls = make_classification(n_samples=500, n_features=n_features,
                                   n_classes=3, n_informative=10,
                                   n_redundant=5, random_state=42)
    X_reg, y_reg = make_regression(n_samples=500, n_features=n_features,
                                   random_state=42)
    y_reg = np.abs(y_reg) % 200 + 10  # قيم SL بين 10 و 210

    # نموذج الاتجاه
    model_dir = xgb.XGBClassifier(n_estimators=50, max_depth=3,
                                   use_label_encoder=False,
                                   eval_metric='mlogloss', random_state=42)
    model_dir.fit(X, y_cls)

    # نموذج SL
    model_sl = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    model_sl.fit(X_reg, y_reg)

    # نموذج TP
    model_tp = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    model_tp.fit(X_reg, y_reg * 2.5)

    pkg = {
        'model_dir'      : model_dir,
        'model_sl'       : model_sl,
        'model_tp'       : model_tp,
        'feature_columns': FEATURE_COLUMNS,
        'config'         : cfg,
        'version'        : '3.0-dummy',
        'asset_key'      : key,
    }

    out = f"models/{key}_model.pkl"
    joblib.dump(pkg, out)
    print(f"  ✅ تم حفظ: {out}")

print("\n🎉 تم إنشاء النماذج الوهمية!")
print("   ⚠️  هذه نماذج للاختبار فقط — شغّل train_multi_model.py لاحقاً للحصول على نماذج حقيقية")
