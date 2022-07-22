
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from predict import prediction
app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == "POST":
        image = request.files.get('imgup')
        image.save('./' + secure_filename(image.filename))
        name, score = prediction(image.filename)
        kwargs = {'name': name, 'score': score}
        return render_template('index2.html', **kwargs)


if __name__ == '__main__':
    app.run()
