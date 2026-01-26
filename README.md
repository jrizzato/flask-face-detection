# Welcome to GenderVision

This is a demo project that shows how to create a web app using Flask and integrating a machine learning model. GenderVision takes a picture as an input and determines the gender of the person in the picture.

***

![](image.png)

![](image-1.png)

![](image-2.png)

***

## Getting Started

### Training Data

This project uses the **IMDB-WIKI dataset** for training. To obtain the training data:

1. Visit: [IMDB-WIKI Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
2. Download "faces only" (1GB)
3. Place the data in the `data/` directory

**Note**: The IMDB-WIKI dataset is provided for research purposes. Please review the dataset's license and terms of use before downloading.

### Setup Instructions

```bash
pip install -r requirements.txt
python modules/01-image_croping.py
python modules/02-structuring_data.py
python modules/03-data_preprocessing.py
python modules/04-eigen_images.py
python modules/05-model.py
```

Then run the flask app
```bash
flask --app server run
```

Once que app is open, navigate to http://127.0.0.1:5000/detect, choose a picture and then click in upload & predict button

## Disclaimer
This is a **demonstration project** for portfolio purposes only. 
It is NOT intended for production use and lacks security hardening.

### Known Limitations:
- No file upload validation
- Debug mode enabled
- No authentication system
- Simplified error handling

## License

This project is provided as-is for educational purposes.

The IMDB-WIKI dataset is used under the terms specified by its creators.