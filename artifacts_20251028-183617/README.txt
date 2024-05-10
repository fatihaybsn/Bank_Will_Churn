Bu klasör, churn ANN eğitimi sonrası üretilen artefaktları içerir.
- Model: FatihAybsn.keras
- Ön-işleme nesneleri: *.pkl (varsa)
- Eğitim geçmişi: history.json ve training_curves.png (varsa)
- Özellik isimleri: feature_names.json (varsa)

Yükleme örneği:
import tensorflow as tf, pickle
model = tf.keras.models.load_model(r'artifacts_20251028-183617\FatihAybsn.keras')
# scaler/ct/encoders için: pickle.load(open('...pkl','rb'))
