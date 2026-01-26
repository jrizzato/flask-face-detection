import cv2
from modules.config import app
from flask import render_template, request
from modules.pipeline_flow import pipeline_flow

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pipeline')
def pipeline():
    return render_template('pipeline.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        f = request.files['image'] # image is name in form
        # print(type(f)) # <class 'werkzeug.datastructures.file_storage.FileStorage'>
        # print(f) # <FileStorage: '11.png' ('image/png')>
        filename = f.filename
        # print(type(filename)) # <class 'str'>
        # print(filename) # 11.png
        path = f'./static/uploads/{filename}'
        # print(type(path)) # <class 'str'>
        # print(path) # ./static/uploads\11.png
        f.save(path)
        img = pipeline_flow(path)
        filename_gender = f'{filename[:-4]}_gender.png'
        cv2.imwrite(f"./static/predict/{filename_gender}", img)

        return render_template('detect.html', upload=True, filename=filename, filename_gender=filename_gender)
    return render_template('detect.html', upload=False)

if __name__ == '__main__':
    # NOTE: debug=True is for development only
    # In production, use: app.run(debug=False, host='0.0.0.0')
    app.run(debug=True)