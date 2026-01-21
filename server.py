from modules.config import app
from flask import render_template

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

if __name__ == '__main__':
    app.run(debug=True)