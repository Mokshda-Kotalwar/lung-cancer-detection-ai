from backend.ml.predictor import predictor
import io
from PIL import Image

img = Image.new('RGB', (224, 224), 'white')
buf = io.BytesIO()
img.save(buf, format='PNG')
data = buf.getvalue()
res = predictor.predict(data)
print(res)
