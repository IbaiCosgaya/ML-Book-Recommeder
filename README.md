---
title: DeLibreroo
emoji: 🐠
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# 📚 DeLibreroo: Sistema de Recomendación de Libros (Filtrado Colaborativo)

Esta aplicación Streamlit presenta un sistema de recomendación de libros basado en **filtrado colaborativo**, utilizando la potente librería `surprise`. El objetivo es ayudar a los usuarios a descubrir nuevos libros basándose en las preferencias de usuarios con gustos similares.

## ✨ Características Principales:

* **Filtrado Colaborativo Basado en Modelos:** Utiliza el algoritmo **Singular Value Decomposition (SVD)** implementado en la librería `surprise` para aprender patrones de las valoraciones de los usuarios y predecir valoraciones para libros no vistos.
* **Modelos Entrenados:** Incorpora modelos `SVD` y `k-NN` (para encontrar usuarios similares) pre-entrenados para recomendaciones robustas.
* **Manejo de "Cold Start" (Usuarios Nuevos):** Para los usuarios que son nuevos en la plataforma y aún no han calificado suficientes libros, la aplicación solicita al menos 3 valoraciones iniciales. Esto permite al modelo construir un perfil básico de preferencias y ofrecer recomendaciones iniciales antes de integrarlo completamente con el modelo colaborativo.
* **Métricas de Evaluación:** La calidad del modelo se evalúa internamente utilizando métricas como la **Precisión y Recall @ K** (Precision@K y Recall@K), asegurando que las recomendaciones son relevantes entre los principales resultados.
* **Interfaz Interactiva:** Construida con Streamlit, ofrece una experiencia de usuario intuitiva para buscar libros, registrar valoraciones y recibir recomendaciones personalizadas.
* **Recomendaciones para Usuarios Existentes y Nuevos:** Permite tanto la simulación de un nuevo usuario que valora libros para obtener recomendaciones, como la selección de usuarios existentes para explorar sus recomendaciones.
* **Exploración de Usuarios Similares:** Permite visualizar los libros calificados por usuarios similares a un usuario objetivo, para entender mejor las recomendaciones.

## ⚙️ Tecnologías Utilizadas:

* **Streamlit:** Para la interfaz de usuario interactiva.
* **Python:** Lenguaje de programación principal.
* **`surprise`:** Librería fundamental para los algoritmos de filtrado colaborativo (SVD, k-NN).
* **Pandas & NumPy:** Para manipulación y análisis de datos.
* **Joblib:** Para la serialización y carga eficiente de modelos.
* **Docker:** Para el empaquetado de la aplicación y sus dependencias, asegurando un entorno consistente de despliegue en Hugging Face Spaces y facilitando futuros despliegues en plataformas como AWS.

## 🚀 Cómo Funciona (en el despliegue):

Esta aplicación está empaquetada en un contenedor Docker. Hugging Face Spaces utiliza el `Dockerfile` proporcionado para construir el entorno necesario, instalar todas las dependencias (incluido `scikit-surprise` en Python 3.9) y ejecutar la aplicación Streamlit.

## 🔗 Acceso al Space:

[El enlace a tu Hugging Face Space aparecerá aquí una vez desplegado]

---
**Autor:** [IbaiCosgaya]
**Repositorio de GitHub:** [[Enlace a tu repositorio de GitHub](https://github.com/IbaiCosgaya)]
