# ARIA SYAF QUANT PRO — Unified Intelligence Server v6.0

خادم الذكاء الاصطناعي لروبوت التداول ARIA. يستقبل بيانات السوق من MT4 ويُرجع قرارات تداول شاملة.

---

## هيكل الملفات

```
├── app.py                  ← الخادم الرئيسي (Flask + XGBoost)
├── train_multi_model.py    ← تدريب النماذج على بياناتك الحقيقية
├── create_dummy_models.py  ← إنشاء نماذج وهمية للاختبار السريع
├── requirements.txt        ← المكتبات المطلوبة
├── Procfile                ← أمر تشغيل Render.com
├── render.yaml             ← إعدادات Render.com التلقائية
├── .gitignore
└── models/
    ├── btc_model.pkl       ← نموذج BTC/Crypto
    ├── gold_model.pkl      ← نموذج XAUUSD/Gold
    └── forex_model.pkl     ← نموذج Forex Majors
```

---

## النشر على Render.com

### الطريقة الأسرع (تلقائي)
1. ارفع هذا المجلد على GitHub
2. في Render.com اختر **New Web Service** → اربطه بالمستودع
3. Render سيقرأ `render.yaml` تلقائياً ويضبط كل شيء
4. اضغط **Deploy**

### التحقق من النجاح
```
GET https://your-app.onrender.com/health
```
الرد المتوقع:
```json
{"status": "ready", "models": {"btc": "BTC/Crypto", "gold": "XAUUSD/Gold", "forex": "Forex Majors"}}
```

---

## تدريب النماذج الحقيقية

```bash
# تدريب على بياناتك (CSV من MT4)
python train_multi_model.py --btc BTCUSD_M1.csv --gold XAUUSD_M1.csv --forex EURUSD_M1.csv

# أو تدريب زوج واحد فقط
python train_multi_model.py --btc BTCUSD_M1.csv
```

بعد التدريب ارفع ملفات `models/*.pkl` المحدّثة على GitHub وأعد النشر.

---

## API Endpoints

| Endpoint | Method | الوظيفة |
|----------|--------|---------|
| `/health` | GET | فحص حالة الخادم والنماذج |
| `/predict` | POST | الحصول على قرار التداول |
| `/detect` | GET | اكتشاف نوع الزوج |

---

## إعداد MT4

في إعدادات الروبوت:
- `InpXGBoostAPIURL` = `https://your-app.onrender.com/predict`
- `InpEnableXGBoost` = `true`
- `InpStealthMode` = `true`
