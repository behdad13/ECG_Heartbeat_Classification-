from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from predict import get_prediction

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediction = get_prediction(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            prediction = 'Prediction is: ' + str(prediction)
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
