# 🐍Flask MAchineLearning Face recognition 

Create a gender classification model ans integrate it to a Flask App.

The task is to develop a Machine Learning Model wich should automatically detect faces ans classify genders.

User will upload an image and the app has to detect the face and identify the gender.

---
## 🏗 Arquitectura General

### Steps:
#### OpenCV
* Read Image
* Gray Scale
* Face Detection

#### numpy
* index, structure and normalize

#### sklearn
* Eigen Image with PCA
* machine learning model

## Machine Model Application Flow
upload Image -> crop image -> data preprocessing -> feature extraction -> ML model -> output

---
## 📑Dependencias

1. **Python 3.x**
2. **matplotlib** (`pip install matplotlib`)
3. Dependencias listadas en requierements.txt

---
## 🚀Cómo Ejecutar el Proyecto
1. **Clonar o descargar** el repositorio.

2. **Crear y activar** un entorno virtual.

3. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
   El archivo `requirements.txt` se encuentran en la carpeta [deps](./deps) del proyecto.

---
## 🙎‍♀️🙎‍♂️Autores

- Apellido y Nombre del primer integrante
- Apellido y Nombre del primer integrante

---

> **Consejo**: Mantén el README **actualizado** conforme evoluciona el proyecto, y elimina (o añade) secciones según necesites. Esta plantilla es sólo un punto de partida general.
