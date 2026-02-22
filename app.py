"""
==============================================================
  ARIA SYAF Adaptive XGBoost Server v3.0
  خادم ذكي يكتشف نوع الزوج تلقائياً ويختار النموذج المناسب
==============================================================
  /predict  ← يستقبل JSON من MT4 ويعيد SL/TP/Direction
  /health   ← فحص حالة الخادم والنماذج المحمّلة
  /models   ← قائمة النماذج المتاحة
==============================================================
"""

from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import re
import logging
import math
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ================================================================
#  تحميل النماذج عند بدء الخادم
# ================================================================
MODELS = {}          # {"btc": pkg, "gold": pkg, "forex": pkg}
MODELS_DIR = os.environ.get("MODELS_DIR", "models")

def load_all_models():
    for key in ["btc", "gold", "forex"]:
        path = os.path.join(MODELS_DIR, f"{key}_model.pkl")
        if os.path.exists(path):
            try:
                MODELS[key] = joblib.load(path)
                cfg = MODELS[key].get('config', {})
                logger.info(f"✅ نموذج {key} محمّل — {cfg.get('name','?')} | {len(MODELS[key]['feature_columns'])} ميزة")
            except Exception as e:
                logger.error(f"❌ فشل تحميل نموذج {key}: {e}")
        else:
            logger.warning(f"⚠️  نموذج {key} غير موجود: {path}")

load_all_models()


# ================================================================
#  اكتشاف نوع الزوج تلقائياً
# ================================================================
BTC_SYMBOLS  = {'BTCUSD','BTCUSDT','XBTUSD','BTC','BTCEUR','BTCGBP','ETHUSD','ETHUSDT','ETH'}
GOLD_SYMBOLS = {'XAUUSD','GOLD','XAUEUR','XAUGBP','XAUJPY','XAUAUD','XAUCHF'}
FOREX_PREFIXES = {'EUR','GBP','USD','JPY','AUD','NZD','CAD','CHF','SGD','HKD','NOK','SEK','DKK','MXN','ZAR','TRY'}

def detect_asset_class(symbol: str) -> str:
    """
    يكتشف فئة الأصل من اسم الزوج:
      - btc   : BTCUSD, ETHUSD, BTCUSDT, ...
      - gold  : XAUUSD, GOLD, ...
      - forex : EURUSD, GBPJPY, USDJPY, ...
    """
    s = symbol.upper().replace("/","").replace("-","").replace("_","").replace(".","")
    # إزالة اللاحقات الشائعة
    for suffix in ['PRO','MICRO','MINI','ECN','STP','RAW','CASH','SPOT']:
        s = s.replace(suffix, '')

    if s in BTC_SYMBOLS or s.startswith('BTC') or s.startswith('ETH') or \
       any(x in s for x in ['BTC','ETH','XRP','LTC','ADA','SOL','DOT','DOGE','MATIC','LINK']):
        return 'btc'

    if s in GOLD_SYMBOLS or s.startswith('XAU') or 'GOLD' in s:
        return 'gold'

    # فوركس: زوجان من العملات الرئيسية
    base = s[:3]
    quote = s[3:6] if len(s) >= 6 else ''
    if base in FOREX_PREFIXES or quote in FOREX_PREFIXES:
        return 'forex'

    # افتراضي: فوركس
    logger.warning(f"⚠️  لم يُتعرَّف على الزوج '{symbol}' — استخدام نموذج forex")
    return 'forex'


# ================================================================
#  بناء متجه الميزات من بيانات MT4
# ================================================================
def build_feature_vector(data: dict, asset_key: str) -> np.ndarray:
    d = data
    asset_map = {"forex": 0.0, "gold": 0.5, "btc": 1.0}

    fast_ema = float(d.get('fast_ema', 0))
    slow_ema = float(d.get('slow_ema', 0))
    ema200   = float(d.get('ema200', 0))
    close    = float(d.get('close1', fast_ema or 1))
    atr14    = float(d.get('atr14', 1)) or 1
    atr7     = float(d.get('atr7', atr14))
    atr21    = float(d.get('atr21', atr14))

    ema_cross        = float(np.sign(fast_ema - slow_ema))
    price_vs_ema200  = (close - ema200) / (ema200 + 1e-10) if ema200 > 0 else 0
    price_vs_fast    = (close - fast_ema) / (fast_ema + 1e-10) if fast_ema > 0 else 0
    atr_ratio        = atr7 / (atr21 + 1e-10)

    adx      = float(d.get('adx', 0))
    di_plus  = float(d.get('di_plus', 0))
    di_minus = float(d.get('di_minus', 0))
    cfg      = MODELS.get(asset_key, {}).get('config', {})
    adx_thresh = cfg.get('adx_threshold', 22)
    adx_above  = 1.0 if adx > adx_thresh else 0.0

    rsi14 = float(d.get('rsi14', 50))
    rsi7  = float(d.get('rsi7', 50))
    rsi_ob = 1.0 if rsi14 > 70 else 0.0
    rsi_os = 1.0 if rsi14 < 30 else 0.0

    macd_main    = float(d.get('macd_main', 0))
    macd_sig     = float(d.get('macd_signal', 0))
    macd_hist    = macd_main - macd_sig
    macd_cross   = float(np.sign(macd_main - macd_sig))

    stoch_k    = float(d.get('stoch_k', 50))
    stoch_d    = float(d.get('stoch_d', 50))
    stoch_cross = float(np.sign(stoch_k - stoch_d))

    bb_width      = float(d.get('bb_width', 0))
    bb_width_norm = bb_width / (close + 1e-10)
    bb_position   = float(d.get('bb_position', 0.5))
    bb_squeeze    = float(d.get('bb_squeeze', 0))

    candle_body       = float(d.get('candle_body', 0))
    candle_upper_wick = float(d.get('candle_upper_wick', 0))
    candle_lower_wick = float(d.get('candle_lower_wick', 0))
    candle_dir        = float(d.get('candle_direction', 0))
    chg1 = float(d.get('close_change1', 0))
    chg2 = float(d.get('close_change2', 0))
    chg3 = float(d.get('close_change3', 0))
    trend_3 = (np.sign(chg1) + np.sign(chg2) + np.sign(chg3)) / 3.0

    price_in_range = float(d.get('price_in_range', 0.5))

    hour_sin = float(d.get('hour_sin', 0))
    hour_cos = float(d.get('hour_cos', 1))
    dow_sin  = float(d.get('dow_sin', 0))
    dow_cos  = float(d.get('dow_cos', 1))

    asset_class = asset_map.get(asset_key, 0.0)

    return np.array([
        fast_ema, slow_ema, ema200, ema_cross, price_vs_ema200, price_vs_fast,
        adx, di_plus, di_minus, adx_above,
        atr14, atr7, atr21, atr_ratio,
        rsi14, rsi7, rsi_ob, rsi_os,
        macd_main, macd_sig, macd_hist, macd_cross,
        stoch_k, stoch_d, stoch_cross,
        bb_width, bb_width_norm, bb_position, bb_squeeze,
        candle_body, candle_upper_wick, candle_lower_wick, candle_dir,
        chg1, chg2, chg3, trend_3,
        price_in_range,
        hour_sin, hour_cos, dow_sin, dow_cos,
        asset_class,
    ], dtype=np.float32).reshape(1, -1)


# ================================================================
#  نقطة النهاية الرئيسية: /predict
# ================================================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    يستقبل JSON من MT4 ويعيد:
    {
      "sl_pips": 120.5,
      "tp_pips": 301.2,
      "confidence": 0.78,
      "direction": "BUY",
      "signal_strength": "STRONG",
      "asset_class": "btc",
      "model_used": "BTC/Crypto"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "لا توجد بيانات JSON"}), 400

        # ── اكتشاف نوع الزوج تلقائياً ──
        symbol     = data.get('symbol', data.get('pair', 'EURUSD'))
        asset_key  = detect_asset_class(symbol)

        logger.info(f"📥 {symbol} → فئة: {asset_key} | ADX={data.get('adx','?')} | dir={data.get('direction','?')}")

        # ── التحقق من وجود النموذج ──
        if asset_key not in MODELS:
            # fallback: استخدم أي نموذج متاح
            if MODELS:
                asset_key = list(MODELS.keys())[0]
                logger.warning(f"⚠️  استخدام نموذج بديل: {asset_key}")
            else:
                return jsonify({"error": "لا توجد نماذج محمّلة — شغّل train_multi_model.py أولاً"}), 503

        pkg       = MODELS[asset_key]
        model_dir = pkg['model_dir']
        model_sl  = pkg['model_sl']
        model_tp  = pkg['model_tp']
        cfg       = pkg.get('config', {})

        # ── بناء متجه الميزات ──
        X = build_feature_vector(data, asset_key)

        # ── التنبؤ ──
        proba      = model_dir.predict_proba(X)[0]   # [SELL, NEUTRAL, BUY]
        pred_class = int(model_dir.predict(X)[0]) - 1  # -1, 0, +1
        direction_map   = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}
        confidence_map  = {-1: proba[0], 0: proba[1], 1: proba[2]}
        confidence = float(confidence_map[pred_class])

        sl_pips = float(model_sl.predict(X)[0])
        tp_pips = float(model_tp.predict(X)[0])

        # ضمان قيم منطقية حسب فئة الأصل
        sl_min = {"btc": 20.0, "gold": 5.0, "forex": 5.0}.get(asset_key, 5.0)
        sl_max = {"btc": 800.0, "gold": 200.0, "forex": 100.0}.get(asset_key, 200.0)
        sl_pips = max(sl_min, min(sl_pips, sl_max))
        tp_pips = max(sl_pips * 1.5, min(tp_pips, sl_pips * 5.0))

        # قوة الإشارة
        if confidence >= 0.75:   strength = "VERY_STRONG"
        elif confidence >= 0.65: strength = "STRONG"
        elif confidence >= 0.55: strength = "MODERATE"
        else:                    strength = "WEAK"

        resp = {
            "sl_pips"       : round(sl_pips, 1),
            "tp_pips"       : round(tp_pips, 1),
            "confidence"    : round(confidence, 4),
            "direction"     : direction_map[pred_class],
            "signal_strength": strength,
            "asset_class"   : asset_key,
            "model_used"    : cfg.get('name', asset_key),
            "proba_buy"     : round(float(proba[2]), 4),
            "proba_sell"    : round(float(proba[0]), 4),
            "proba_neutral" : round(float(proba[1]), 4),
        }

        logger.info(f"📤 {direction_map[pred_class]} | conf={confidence:.2f} | SL={sl_pips:.0f} | TP={tp_pips:.0f} | [{asset_key}]")
        return jsonify(resp)

    except Exception as e:
        logger.error(f"❌ خطأ: {e}")
        return jsonify({"error": str(e)}), 500


# ================================================================
#  نقطة فحص الصحة: /health
# ================================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status" : "ready" if MODELS else "no_models",
        "models" : {k: v.get('config',{}).get('name','?') for k, v in MODELS.items()},
        "version": "3.0",
    })


# ================================================================
#  قائمة النماذج: /models
# ================================================================
@app.route('/models', methods=['GET'])
def list_models():
    info = {}
    for k, v in MODELS.items():
        cfg = v.get('config', {})
        info[k] = {
            "name"       : cfg.get('name', k),
            "description": cfg.get('description', ''),
            "features"   : len(v.get('feature_columns', [])),
            "adx_thresh" : cfg.get('adx_threshold', '?'),
        }
    return jsonify(info)


# ================================================================
#  اختبار اكتشاف الزوج: /detect?symbol=BTCUSD
# ================================================================
@app.route('/detect', methods=['GET'])
def detect():
    symbol = request.args.get('symbol', 'EURUSD')
    asset  = detect_asset_class(symbol)
    return jsonify({"symbol": symbol, "detected_class": asset,
                    "model_available": asset in MODELS})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 ARIA Adaptive Server v3.0 — منفذ {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

# ================================================================
#  WSGI entry point للـ gunicorn
#  Start Command: gunicorn app:app --workers 1 --timeout 120
# ================================================================
