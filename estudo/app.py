

from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import io, os, base64

# Configura Flask
app = Flask(__name__)

# --- Carrega o modelo e ready ---
model = torch.load('model.pth', map_location='cpu', weights_only=False)
model.eval()

# Transformação de pré-processamento
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # checa upload
    if 'file' not in request.files:
        return render_template('index.html', error='Nenhum arquivo enviado.')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Arquivo inválido.')

    # lê imagem
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    out = model(tensor)
    _, pred = torch.max(out, 1)
    # pega nome da classe
    label = ['A','B','C','D','E','F','G','I','L','M','N','O','P','Q','R','S','T','U','V','W','Y'][pred.item()]

    # converte imagem para base64
    buff = io.BytesIO()
    img.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue()).decode()

    return render_template('index.html', prediction=label, img_data=img_str)

if __name__ == '__main__':
    # debug=True recarrega ao editar o template
    app.run(host='0.0.0.0', port=5000, debug=True)

