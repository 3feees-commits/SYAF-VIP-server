"""
ARIA SYAF QUANT PRO — Unified Intelligence Server v4.0
=======================================================
يستقبل كل بيانات الروبوت ويُرجع قرارات شاملة:
  - إشارة الدخول (BUY/SELL/NONE)
  - SL/TP محسوب بالنقاط
  - مستوى Breakeven الذكي
  - خطوة Trailing المثلى
  - حجم اللوت المقترح
  - تقييم المخاطر
  - ثقة النموذج
"""

from flask import Flask, request, jsonify
import pickle, os, logging, math
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

MODELS_DIR = "models"

# ===================================================================
#  تحميل النماذج
# ===================================================================
models = {}
scalers = {}
feature_cols = {}

def load_models():
    classes = {"btc": "BTC/Crypto", "gold": "XAUUSD/Gold", "forex": "Forex Majors"}
    for cls in classes:
        mpath = os.path.join(MODELS_DIR, f"{cls}_model.pkl")
        spath = os.path.join(MODELS_DIR, f"{cls}_scaler.pkl")
        fpath = os.path.join(MODELS_DIR, f"{cls}_features.pkl")
        if os.path.exists(mpath):
            with open(mpath, "rb") as f: models[cls] = pickle.load(f)
            logging.info(f"✅ نموذج {cls} محمّل")
        if os.path.exists(spath):
            with open(spath, "rb") as f: scalers[cls] = pickle.load(f)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f: feature_cols[cls] = pickle.load(f)

load_models()

# ===================================================================
#  اكتشاف نوع الزوج
# ===================================================================
def detect_class(symbol: str) -> str:
    s = symbol.upper()
    if any(x in s for x in ["BTC","ETH","XRP","LTC","ADA","SOL","DOGE","BNB","DOT","LINK"]):
        return "btc"
    if any(x in s for x in ["XAU","GOLD","XAG","SILVER"]):
        return "gold"
    return "forex"

# ===================================================================
#  حساب التكاليف بالنقاط (يطابق منطق MT4)
# ===================================================================
def calc_costs_pips(spread_pips: float, commission_per_001: float, lots: float) -> float:
    comm_total = commission_per_001 * (lots / 0.01)
    # تقريب: العمولة ≈ نصف السبريد كمسافة سعرية
    comm_pips = comm_total * 2  # تقدير تحويلي
    return spread_pips + comm_pips

# ===================================================================
#  حساب حجم اللوت المقترح بناءً على المخاطرة
# ===================================================================
def suggest_lot(balance: float, risk_pct: float, sl_pips: float,
                pip_value_per_001: float, base_lot: float = 0.01) -> float:
    if sl_pips <= 0 or pip_value_per_001 <= 0:
        return base_lot
    risk_amount = balance * (risk_pct / 100.0)
    sl_cost_per_001 = sl_pips * pip_value_per_001
    if sl_cost_per_001 <= 0:
        return base_lot
    suggested = (risk_amount / sl_cost_per_001) * 0.01
    suggested = max(0.01, min(suggested, 100.0))
    suggested = round(suggested / 0.01) * 0.01
    return round(suggested, 2)

# ===================================================================
#  تقييم المخاطر
# ===================================================================
def assess_risk(adx: float, spread_pips: float, symbol_class: str,
                confidence: float, atr: float) -> dict:
    risk_score = 0
    warnings = []

    # ADX
    if adx < 20:
        risk_score += 3
        warnings.append("ADX ضعيف — سوق عرضي")
    elif adx < 25:
        risk_score += 1

    # Spread
    max_spreads = {"btc": 50, "gold": 15, "forex": 3}
    max_sp = max_spreads.get(symbol_class, 3)
    if spread_pips > max_sp * 2:
        risk_score += 3
        warnings.append(f"سبريد مرتفع جداً ({spread_pips:.1f})")
    elif spread_pips > max_sp:
        risk_score += 1
        warnings.append(f"سبريد مرتفع ({spread_pips:.1f})")

    # Confidence
    if confidence < 0.55:
        risk_score += 3
        warnings.append("ثقة النموذج منخفضة")
    elif confidence < 0.65:
        risk_score += 1

    # ATR
    if atr <= 0:
        risk_score += 2
        warnings.append("ATR غير متاح")

    # تقييم نهائي
    if risk_score >= 5:
        level = "HIGH"
        action = "AVOID"
    elif risk_score >= 3:
        level = "MEDIUM"
        action = "REDUCE_LOT"
    else:
        level = "LOW"
        action = "PROCEED"

    return {"level": level, "score": risk_score, "action": action, "warnings": warnings}

# ===================================================================
#  الحساب الأساسي للـ SL/TP/BE/Trail
# ===================================================================
def calculate_targets(symbol_class: str, direction: int, atr: float,
                      spread_pips: float, commission_per_001: float,
                      lots: float, sl_mult: float, tp_mult: float,
                      net_profit_per_001: float, confidence: float) -> dict:

    # تكاليف الصفقة
    costs_pips = calc_costs_pips(spread_pips, commission_per_001, lots)

    # SL بناءً على ATR
    sl_pips = max(atr * sl_mult, 5.0)

    # Net TP = SL × TP_Mult + تكاليف (لضمان الربح الصافي)
    tp_pips = sl_pips * tp_mult + costs_pips

    # Smart Breakeven = تكاليف + ربح صافي مستهدف
    net_profit_pips = net_profit_per_001 * (lots / 0.01) * 2  # تقدير تحويلي
    be_pips = costs_pips + net_profit_pips

    # Trailing Step = 40% من ATR (يبدأ بعد تجاوز التكاليف)
    trail_pips = max(atr * 0.4, costs_pips + 2)

    # تعديل بناءً على الثقة
    if confidence >= 0.80:
        tp_pips *= 1.2   # هدف أكبر عند ثقة عالية
        trail_pips *= 1.1
    elif confidence < 0.60:
        sl_pips *= 0.8   # SL أضيق عند ثقة منخفضة
        tp_pips *= 0.9

    return {
        "sl_pips":    round(sl_pips, 2),
        "tp_pips":    round(tp_pips, 2),
        "be_pips":    round(be_pips, 2),
        "trail_pips": round(trail_pips, 2),
        "costs_pips": round(costs_pips, 2),
    }

# ===================================================================
#  الـ Endpoint الرئيسي الموحّد /predict
# ===================================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "لا توجد بيانات"}), 400

        # ── استخراج البيانات الأساسية ──
        symbol        = data.get("symbol", "UNKNOWN")
        direction     = int(data.get("direction", 0))
        fast_ema      = float(data.get("fast_ema", 0))
        slow_ema      = float(data.get("slow_ema", 0))
        adx           = float(data.get("adx", 0))
        di_plus       = float(data.get("di_plus", 0))
        di_minus      = float(data.get("di_minus", 0))
        atr           = float(data.get("atr", 0))
        rsi           = float(data.get("rsi", 50))
        close1        = float(data.get("close1", 0))
        close2        = float(data.get("close2", 0))
        close3        = float(data.get("close3", 0))
        spread        = float(data.get("spread", 2))

        # ── بيانات إدارة المخاطر (جديدة) ──
        account_balance     = float(data.get("account_balance", 10000))
        lots                = float(data.get("lots", 0.01))
        risk_percent        = float(data.get("risk_percent", 1.0))
        commission_per_001  = float(data.get("commission_per_001", 0.15))
        net_profit_per_001  = float(data.get("net_profit_per_001", 0.30))
        pip_value_per_001   = float(data.get("pip_value_per_001", 0.10))
        sl_mult             = float(data.get("sl_mult", 1.0))
        tp_mult             = float(data.get("tp_mult", 2.5))

        # ── بيانات الصفقة المفتوحة (للإدارة) ──
        open_ticket         = int(data.get("open_ticket", -1))
        open_price          = float(data.get("open_price", 0))
        open_type           = int(data.get("open_type", -1))   # 0=BUY, 1=SELL
        current_profit      = float(data.get("current_profit", 0))
        virtual_sl          = float(data.get("virtual_sl", 0))
        virtual_tp          = float(data.get("virtual_tp", 0))
        be_done             = bool(data.get("be_done", False))

        # ── بيانات XGBoost ──
        lr      = float(data.get("learning_rate", 0.1))
        depth   = int(data.get("max_depth", 8))
        subsamp = float(data.get("subsample", 0.9))

        # ── اكتشاف نوع الزوج ──
        sym_class = detect_class(symbol)
        model     = models.get(sym_class)

        # ── تشغيل XGBoost ──
        confidence = 0.60  # افتراضي
        xgb_signal = direction  # افتراضي: نفس إشارة MT4

        if model is not None:
            try:
                import numpy as np
                hour = datetime.now().hour
                dow  = datetime.now().weekday()
                asset_class = {"btc": 1.0, "gold": 0.5, "forex": 0.0}[sym_class]

                feat = [fast_ema, slow_ema, adx, di_plus, di_minus, atr, rsi,
                        close1, close2, close3, spread, direction,
                        fast_ema - slow_ema,           # EMA crossover
                        adx * direction,               # ADX × direction
                        rsi - 50,                      # RSI deviation
                        (close1 - close2) / (atr + 1e-9),  # momentum
                        math.sin(2 * math.pi * hour / 24),  # hour sin
                        math.cos(2 * math.pi * hour / 24),  # hour cos
                        math.sin(2 * math.pi * dow / 7),    # day sin
                        math.cos(2 * math.pi * dow / 7),    # day cos
                        asset_class, lr, depth, subsamp,
                        account_balance, lots, commission_per_001,
                        spread / (atr + 1e-9),         # spread/ATR ratio
                        current_profit]

                X = np.array([feat])
                sc = scalers.get(sym_class)
                if sc:
                    try: X = sc.transform(X)
                    except: pass

                prob = model.predict_proba(X)[0]
                confidence = float(max(prob))
                xgb_signal = int(model.predict(X)[0])
                logging.info(f"XGBoost {sym_class}: signal={xgb_signal} conf={confidence:.2f}")
            except Exception as e:
                logging.warning(f"XGBoost error: {e}")

        # ── حساب الأهداف الشاملة ──
        targets = calculate_targets(
            sym_class, direction, atr, spread,
            commission_per_001, lots, sl_mult, tp_mult,
            net_profit_per_001, confidence
        )

        # ── تقييم المخاطر ──
        risk = assess_risk(adx, spread, sym_class, confidence, atr)

        # ── قرار الدخول النهائي ──
        final_signal = "NONE"
        if risk["action"] == "AVOID":
            final_signal = "NONE"
        elif confidence >= 0.55 and direction != 0:
            if xgb_signal == direction or model is None:
                final_signal = "BUY" if direction == 1 else "SELL"
            else:
                final_signal = "NONE"  # XGBoost يعارض إشارة MT4

        # ── إدارة الصفقة المفتوحة ──
        trade_management = {}
        if open_ticket > 0 and open_price > 0 and open_type >= 0:
            current_bid = close1
            costs = targets["costs_pips"]
            net_pips = net_profit_per_001 * (lots / 0.01) * 2

            # هل يجب تحريك BE؟
            if not be_done:
                smart_target = costs + net_pips
                if open_type == 0:  # BUY
                    be_trigger = open_price + (smart_target + 20) * atr / adx if adx > 0 else open_price + smart_target * 0.0001
                    new_be = open_price + smart_target * 0.0001
                    if current_bid >= be_trigger:
                        trade_management["action"] = "MOVE_BE"
                        trade_management["new_virtual_sl"] = round(new_be, 5)
                        trade_management["reason"] = "Smart BE: تغطية التكاليف + ربح صافي"
                else:  # SELL
                    be_trigger = open_price - (smart_target + 20) * atr / adx if adx > 0 else open_price - smart_target * 0.0001
                    new_be = open_price - smart_target * 0.0001
                    if current_bid <= be_trigger:
                        trade_management["action"] = "MOVE_BE"
                        trade_management["new_virtual_sl"] = round(new_be, 5)
                        trade_management["reason"] = "Smart BE: تغطية التكاليف + ربح صافي"

            # هل يجب تحريك Trailing؟
            if "action" not in trade_management:
                trail = targets["trail_pips"] * 0.0001
                if open_type == 0 and current_bid - open_price > trail + costs * 0.0001:
                    new_sl = current_bid - trail
                    if virtual_sl == 0 or new_sl > virtual_sl:
                        trade_management["action"] = "MOVE_TRAIL"
                        trade_management["new_virtual_sl"] = round(new_sl, 5)
                        trade_management["reason"] = "Trailing: تحرّك السعر"
                elif open_type == 1 and open_price - current_bid > trail + costs * 0.0001:
                    new_sl = current_bid + trail
                    if virtual_sl == 0 or new_sl < virtual_sl:
                        trade_management["action"] = "MOVE_TRAIL"
                        trade_management["new_virtual_sl"] = round(new_sl, 5)
                        trade_management["reason"] = "Trailing: تحرّك السعر"

            if not trade_management:
                trade_management["action"] = "HOLD"
                trade_management["reason"] = "لا تغيير مطلوب"

        # ── حجم اللوت المقترح ──
        suggested_lot = suggest_lot(account_balance, risk_percent,
                                    targets["sl_pips"], pip_value_per_001)
        if risk["action"] == "REDUCE_LOT":
            suggested_lot = round(suggested_lot * 0.5, 2)
            suggested_lot = max(0.01, suggested_lot)

        # ── قوة الإشارة ──
        if confidence >= 0.80:   signal_strength = "VERY_STRONG"
        elif confidence >= 0.70: signal_strength = "STRONG"
        elif confidence >= 0.60: signal_strength = "MODERATE"
        else:                    signal_strength = "WEAK"

        # ── الرد الشامل ──
        response = {
            # === قرار الدخول ===
            "signal":           final_signal,
            "direction":        direction,
            "xgb_signal":       xgb_signal,
            "confidence":       round(confidence, 4),
            "signal_strength":  signal_strength,

            # === أهداف السعر ===
            "sl_pips":          targets["sl_pips"],
            "tp_pips":          targets["tp_pips"],
            "be_pips":          targets["be_pips"],
            "trail_pips":       targets["trail_pips"],
            "costs_pips":       targets["costs_pips"],

            # === إدارة اللوت ===
            "suggested_lot":    suggested_lot,
            "risk_action":      risk["action"],

            # === تقييم المخاطر ===
            "risk_level":       risk["level"],
            "risk_score":       risk["score"],
            "risk_warnings":    risk["warnings"],

            # === إدارة الصفقة المفتوحة ===
            "trade_management": trade_management,

            # === معلومات إضافية ===
            "symbol_class":     sym_class,
            "server_version":   "4.0",
            "timestamp":        datetime.now().isoformat()
        }

        logging.info(f"✅ {symbol} | {final_signal} | conf={confidence:.2f} | SL={targets['sl_pips']} TP={targets['tp_pips']}")
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"❌ خطأ: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ===================================================================
#  Endpoint للصحة
# ===================================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ready",
        "models":  {k: "loaded" for k in models},
        "version": "4.0",
        "endpoints": {
            "/predict": "POST — الرئيسي: إشارة + SL/TP + BE + Trail + مخاطر + لوت",
            "/health":  "GET  — فحص الخادم",
            "/detect":  "GET  — اكتشاف نوع الزوج"
        }
    })

@app.route("/detect", methods=["GET"])
def detect():
    symbol = request.args.get("symbol", "")
    cls    = detect_class(symbol)
    return jsonify({
        "symbol":          symbol,
        "detected_class":  cls,
        "model_available": cls in models
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
