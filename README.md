# BookRecommender 📚✨
### Tu compañero inteligente para explorar el universo de los libros y descubrir tu próxima lectura favorita.

---

## 📖 Introducción

En el inmenso mar de la literatura, hallar el libro ideal puede resultar agobiante. **Book Recommender** es un proyecto de sistema de recomendación sofisticado, concebido para guiarte por esta vastedad y vincular a los lectores con las historias que verdaderamente disfrutarán.

Mediante el uso del **aprendizaje automático**, este sistema no solo propone libros de popularidad, sino que también aprende de tus preferencias y las de usuarios con gustos parecidos para brindarte sugerencias genuinamente personalizadas.

---

## 🚀 ¡Prueba la aplicación aquí!

Puedes probar la aplicación en vivo aquí:
[https://ml-book-recommeder.streamlit.app/](https://ml-book-recommeder.streamlit.app/)


---

## 🌟 Características Principales

* **Recomendaciones Personalizadas:** Obtén sugerencias de libros hechas a tu medida.
* **Perfil de Usuario Interactivo:** Crea tu perfil virtual (edad, género, educación, país) y califica libros fácilmente a través de una interfaz intuitiva.
* **Motor de Recomendación Híbrido Inteligente:** Combina lo mejor de dos mundos para ofrecerte la máxima precisión.
* **Explora Usuarios Similares:** Descubre qué libros han disfrutado otros usuarios con gustos parecidos a los tuyos.
* **Predicción de Calificación:** Averigua qué puntuación (estimación) podrías darle a un libro antes de leerlo.

---

## 🧠 ¿Cómo Funciona? (La Magia por Dentro)

Nuestro sistema de recomendación se basa en una potente estrategia **híbrida** que fusiona dos enfoques complementarios:

1.  **Filtrado Colaborativo (FBC)**:
    * **¿Qué hace?** Analiza las **interacciones pasadas** de los usuarios con los ítems (libros). Busca patrones como: "Usuarios que calificaron este libro de forma similar también les gustó X" o "Este libro es calificado de forma similar por usuarios que califican otros libros de manera parecida". Modelos como **SVD** son excelentes aquí, encontrando relaciones ocultas entre usuarios y libros basándose puramente en las calificaciones.
    * **Fortaleza:** Descubre gustos complejos y puede recomendar libros inesperados que no son directamente similares a lo que has leído, pero que les gustan a personas con gustos parecidos a los tuyos.
    * **Debilidad:** Sufre el "problema del arranque en frío" para **nuevos libros** (si un libro no tiene calificaciones, el modelo no sabe nada de él) y para **nuevos usuarios** (si un usuario no ha calificado nada, no hay base para compararlo).

2.  **Filtrado Basado en Contenido (FBCo)**:
    * **¿Qué hace?** Se centra en las **características de los ítems** (el "contenido" del libro: género, autor, número de páginas, palabras clave de la descripción, etc.) y en el perfil de preferencias del usuario basado en esas características. Si te gustó un libro de fantasía de un autor específico, te recomendará otros libros de fantasía o del mismo autor.
    * **Fortaleza:** Excelente para el "arranque en frío" de **nuevos libros** (si conocemos sus características, podemos recomendarlo) y para **nuevos usuarios** (si el usuario indica algunas preferencias iniciales). Las recomendaciones son fácilmente explicables.
    * **Debilidad:** Puede ser menos sorprendente, ya que tiende a recomendar ítems muy similares a los que ya te gustaron. No descubre gustos nuevos o complejos.

### La Sinergia Híbrida:
Al combinar SVD con el FBCo, mitigamos las debilidades individuales. El FBCo nos ayuda con el arranque en frío, mientras que el SVD nos permite descubrir patrones y recomendaciones más sorprendentes y profundas basadas en la inteligencia colectiva.

---

## 🚧 Desafíos Abordados

Durante el desarrollo, nos enfrentamos a retos comunes en sistemas de recomendación y aplicamos soluciones lógicas:

* **Problema del Arranque en Frío:** ¿Cómo recomendar a un nuevo usuario sin historial de calificaciones o sugerir un libro recién añadido? Nuestro enfoque híbrido utiliza las características de contenido y las primeras interacciones para superar este obstáculo.
* **Discrepancia de Datos (`book_id`):** Manejamos la situación real donde muchos `book_id`s en las calificaciones no tenían metadatos completos en nuestra biblioteca de libros. Nuestro sistema está diseñado para enriquecer y fusionar datos, asegurando que las recomendaciones finales siempre muestren información útil (título, autor, género, etc.), priorizando la calidad de la información para el usuario.

---

## 🚀 Uso e Interfaz Interactiva

Este proyecto está diseñado para ejecutarse en un entorno **Jupyter Notebook**, lo que te permite interactuar directamente con el sistema.

1.  **Clona este repositorio.**
2.  **Abre el Jupyter Notebook principal.**
3.  **Ejecuta las celdas secuencialmente:** Sigue los pasos para crear tu perfil, calificar algunos libros, y luego ¡genera tus recomendaciones personalizadas al instante! Podrás ver los resultados y explorar los libros que tus usuarios similares han disfrutado.

---

## 🛠️ Tecnologías Utilizadas

* **Python:** El lenguaje de programación principal.
* **Pandas:** Para manipulación y análisis de datos.
* **Surprise:** Biblioteca especializada en sistemas de recomendación (para SVD).
* **Scikit-learn:** Para algoritmos de Machine Learning (como KNN).
* **Jupyter Notebook:** Para la interfaz interactiva y el desarrollo.
* **ipywidgets:** Para elementos interactivos en el notebook.
* **HTML/CSS:** Para una presentación atractiva de las recomendaciones.
* **Streamlit:** Para el despliegue de la aplicación web interactiva.

---

## 📈 Futuras Mejoras

* Implementación de características de contenido más sofisticadas (ej., procesamiento de lenguaje natural en descripciones de libros).
* Exploración de otros algoritmos híbridos más complejos.

---

## 📧 Contacto

[IbaiCosgaya] - [ibaicosgayaprieto@gmail.com]

---
