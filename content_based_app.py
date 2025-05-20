# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import os # Para verificar la existencia de archivos

# # --- Cargar datos y modelos (¡Solo se cargan una vez!) ---
# @st.cache_resource
# def load_resources():
#     try:
#         # Asegúrate de que este sea el nombre correcto de tu archivo
#         df_combined_books = pd.read_parquet('df_combined_books_final.parquet')
        
#         with open('tfidf_vectorizer.pkl', 'rb') as f:
#             tfidf_model_loaded = pickle.load(f)
        
#         with open('cosine_sim_matrix.pkl', 'rb') as f:
#             cosine_sim_matrix_loaded = pickle.load(f)
        
#         # Re-crear 'features_df' para asegurar que la indexación funcione correctamente
#         # Debe ser exactamente igual a como se creó al guardar los modelos.
#         features_df = df_combined_books[['title','authors','average_rating', 'genre', 'pages']].astype(str)
#         features_df['title'] = features_df['title'].str.replace(' ', '').str.lower()
#         features_df['authors'] = features_df['authors'].str.replace(' ', '').str.lower()
#         features_df['average_rating'] = features_df['average_rating'].str.replace(' ', '').str.lower()
#         features_df['genre'] = features_df['genre'].str.replace(' ', '').str.lower()
#         features_df['pages'] = features_df['pages'].str.replace(' ', '').str.lower()
#         features_df['combined_features'] = features_df['title'] + ' ' + \
#                                            features_df['authors'] + ' ' + \
#                                            features_df['average_rating'] + ' ' + \
#                                            features_df['genre'] + ' ' + \
#                                            features_df['pages']

#         return df_combined_books, tfidf_model_loaded, cosine_sim_matrix_loaded, features_df
    
#     except FileNotFoundError as e:
#         st.error(f"Error: No se encontraron los archivos de recursos necesarios. Asegúrate de que 'df_combined_books_final.parquet', 'tfidf_vectorizer.pkl' y 'cosine_sim_matrix.pkl' estén en la misma carpeta que 'content_based_app.py'.")
#         st.stop() # Detiene la ejecución de la app si faltan archivos
#     except Exception as e:
#         st.error(f"Ocurrió un error inesperado al cargar los recursos: {e}")
#         st.stop()


# # Carga los recursos al inicio de la aplicación
# df_books, tfidf_model, cosine_sim_matrix_precomputed, features_df_for_indexing = load_resources()


# # --- Función de Recomendación Adaptada ---
# def recommend_books_content_based(title, df_books, tfidf_vec, cosine_sim_mtx, features_df_idx, num_recommendations=5):
#     # Asegúrate de que el título del input se procese igual que en 'features_df_idx'
#     processed_title = title.lower().replace(' ', '')
    
#     try:
#         # Busca el índice del libro seleccionado en el DataFrame de features procesadas
#         idx = features_df_idx[features_df_idx['title'] == processed_title].index[0]
#     except IndexError:
#         st.warning(f"El libro '{title}' no se encontró en nuestra base de datos para la búsqueda de similitud. Por favor, selecciona un libro de la lista.")
#         return pd.DataFrame()

#     sim_scores = list(enumerate(cosine_sim_mtx[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     recommended_indices = []
#     seen_book_ids = set()
    
#     # Obtener el book_id del libro original para excluirlo
#     original_book_id = df_books.iloc[idx]['book_id']

#     for i, score in sim_scores:
#         current_book_id = df_books.iloc[i]['book_id']
#         # Excluir el libro original y evitar duplicados por book_id
#         if current_book_id != original_book_id and current_book_id not in seen_book_ids:
#             recommended_indices.append(i)
#             seen_book_ids.add(current_book_id)
#             if len(recommended_indices) >= num_recommendations: # Limitar a 'num_recommendations'
#                 break
    
#     if not recommended_indices:
#         return pd.DataFrame() # No se encontraron recomendaciones distintas

#     return df_books.iloc[recommended_indices]


# # --- Configuración de Estilo para la Aplicación Streamlit ---
# # No cambian, son las mismas que ya tenías y funcionan bien.
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background-color: #f7eecd; /* Color de fondo general */
#         font-family: 'Roboto', 'Open Sans', 'Segoe UI', 'Arial', sans-serif;
#         color: #333;
#     }
#     .main .block-container {
#         padding-top: 30px;
#         padding-bottom: 30px;
#     }
#     .stSelectbox, .stTextInput, .stSlider { /* Agregado .stSlider */
#         font-size: 1.1em;
#     }
#     .stButton > button {
#         background-color: #4CAF50;
#         color: white;
#         padding: 10px 20px;
#         border-radius: 5px;
#         border: none;
#         cursor: pointer;
#         font-size: 1.1em;
#         font-weight: bold;
#     }
#     .stButton > button:hover {
#         background-color: #45a049;
#     }
#     .book-card-container {
#         display: flex;
#         flex-wrap: wrap;
#         gap: 15px;
#         padding: 15px;
#         border-radius: 8px;
#         background-color: #fff;
#         border: 1px solid #ddd;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.05);
#     }
#     .book-card {
#         flex: 0 0 auto;
#         width: 160px;
#         text-align: center;
#         overflow-wrap: break-word;
#         border: 1px solid #eee;
#         padding: 10px;
#         border-radius: 5px;
#         background-color: #fcfcfc;
#         box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
#         font-size: 10px;
#         display: flex;
#         flex-direction: column;
#         justify-content: space-between;
#         height: 350px;
#     }
#     .book-card img {
#         display: block;
#         margin: 0 auto;
#         border: 1px solid #ccc;
#         height: 150px;
#         width: auto;
#         max-width: 100%;
#         object-fit: contain;
#     }
#     .book-title {
#         font-size: 1em;
#         font-weight: bold;
#         margin-bottom: 5px;
#         height: 3em;
#         overflow: hidden;
#         text-overflow: ellipsis;
#         display: -webkit-box;
#         -webkit-line-clamp: 3;
#         -webkit-box-orient: vertical;
#         color: #333;
#     }
#     .book-details {
#         font-size: 0.85em;
#         margin-top: 5px;
#         color: #555;
#     }
#     .no-cover {
#         width: 100px;
#         height: 150px;
#         border: 1px solid #ccc;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-size: 10px;
#         background-color: #f0f0f0;
#         margin: 0 auto;
#         text-align: center;
#         padding: 5px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # --- Interfaz de Usuario en Streamlit ---
# st.title("Book Recommender 📚")
# st.markdown("### ¡Encuentra tu próxima lectura favorita!")

# # --- Contenedor de filtros ---
# st.sidebar.header("Filtra para encontrar un libro")

# # Campo de texto para buscar por título (funciona como un filtro)
# search_title = st.sidebar.text_input("Buscar por título (parcial):", "").strip()

# # Filtro por Género
# all_genres = [''] + sorted(df_books['genre'].dropna().unique().tolist())
# selected_genre = st.sidebar.selectbox("Filtrar por Género:", all_genres)

# # Filtro por Autor (búsqueda parcial)
# search_author = st.sidebar.text_input("Buscar por Autor (parcial):", "").strip()

# # Filtro por Páginas (usamos los valores min/max del DataFrame)
# min_pages_df = int(df_books['pages'].dropna().min()) if not df_books['pages'].dropna().empty else 0
# max_pages_df = int(df_books['pages'].dropna().max()) if not df_books['pages'].dropna().empty else 1000

# # Asegurarse de que el rango de páginas no sea negativo
# if min_pages_df > max_pages_df:
#     min_pages_df = 0
#     max_pages_df = 1000 # Valor por defecto si hay un problema

# page_range = st.sidebar.slider(
#     "Filtrar por rango de Páginas:",
#     min_value=min_pages_df,
#     max_value=max_pages_df,
#     value=(min_pages_df, max_pages_df)
# )

# # --- Aplicar filtros para la lista de selección ---
# filtered_df_for_selection = df_books.copy()

# if search_title:
#     # Usamos .str.contains con case=False para búsqueda insensible a mayúsculas/minúsculas
#     filtered_df_for_selection = filtered_df_for_selection[
#         filtered_df_for_selection['title'].str.contains(search_title, case=False, na=False)
#     ]

# if selected_genre and selected_genre != '':
#     filtered_df_for_selection = filtered_df_for_selection[
#         filtered_df_for_selection['genre'].str.lower() == selected_genre.lower()
#     ]

# if search_author:
#     filtered_df_for_selection = filtered_df_for_selection[
#         filtered_df_for_selection['authors'].str.contains(search_author, case=False, na=False)
#     ]

# # Filtrar por rango de páginas (convertir la columna 'pages' a numérico antes de filtrar)
# try:
#     # Manejar posibles errores de conversión y NaN antes de filtrar
#     filtered_df_for_selection['pages_numeric'] = pd.to_numeric(filtered_df_for_selection['pages'], errors='coerce')
#     filtered_df_for_selection = filtered_df_for_selection[
#         (filtered_df_for_selection['pages_numeric'] >= page_range[0]) & 
#         (filtered_df_for_selection['pages_numeric'] <= page_range[1])
#     ].drop(columns='pages_numeric') # Eliminar la columna temporal
# except Exception as e:
#     st.sidebar.warning(f"Error al filtrar por páginas: {e}. Asegúrate de que la columna 'pages' es numérica.")
#     # Si hay un error, no aplicamos este filtro para no detener la app

# # Obtener la lista de títulos únicos de los libros filtrados
# available_book_titles = filtered_df_for_selection['title'].unique().tolist()
# available_book_titles_sorted = sorted(available_book_titles)

# # Selectbox para seleccionar el libro (ahora se llena con los libros filtrados)
# selected_book_to_recommend = st.selectbox(
#     "Selecciona un libro para encontrar recomendaciones similares:",
#     options=[''] + available_book_titles_sorted
# )

# if selected_book_to_recommend and selected_book_to_recommend != '':
#     if st.button(f"🔎 Buscar recomendaciones para '{selected_book_to_recommend}'"):
#         with st.spinner("Buscando recomendaciones..."):
#             recommendations = recommend_books_content_based(
#                 selected_book_to_recommend,
#                 df_books,
#                 tfidf_model,
#                 cosine_sim_matrix_precomputed,
#                 features_df_for_indexing,
#                 num_recommendations=5 # Asegúrate de que este valor esté en 5
#                     )

#             if not recommendations.empty:
#                 st.markdown(f"### Libros similares a **{selected_book_to_recommend}**:")
                
#                 # Generar el HTML para mostrar las tarjetas de libros
#                 # ¡Este HTML se renderiza correctamente gracias a unsafe_allow_html=True!
#                 html_output = "<div class='book-card-container'>"
#                 for index, book in recommendations.iterrows():
#                     title = book.get('title', "Título desconocido")
#                     author = book.get('authors', "Autor desconocido")
#                     genre = book.get('genre', "Género desconocido")
#                     pages = book.get('pages', "Páginas desconocidas")
#                     image_url = book.get('image_url')

#                     pages_str = "Páginas desconocidas"
#                     try:
#                         if pd.notna(pages):
#                             if pd.api.types.is_numeric_dtype(type(pages)) and not pd.isna(pages):
#                                 pages_str = f"{int(pages)} págs."
#                             else:
#                                 pages_str = str(pages)
#                     except (ValueError, TypeError):
#                         pages_str = str(pages)

#                     html_output += f"""
#                     <div class="book-card">
#                         <p class="book-title">{title}</p>
#                         {f'<img src="{image_url}" alt="Portada de {title}">' if pd.notna(image_url) and image_url and isinstance(image_url, str) else '<div class="no-cover">No hay<br>portada</div>'}
#                         <div class="book-details">
#                             <p>Autor: {author}</p>
#                             <p>Género: {genre}</p>
#                             <p>{pages_str}</p>
#                         </div>
#                     </div>
#                     """
#                 html_output += "</div>"
#                 st.markdown(html_output, unsafe_allow_html=True) # Aquí se renderiza como HTML
#             else:
#                 st.warning(f"No se encontraron recomendaciones similares para '{selected_book_to_recommend}'.")
# else:
#     st.info("Utiliza los filtros de la izquierda para acotar tu búsqueda, o selecciona un libro del menú desplegable.")

import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- Cargar datos y modelos (¡Solo se cargan una vez!) ---
@st.cache_resource
def load_resources():
    try:
        df_combined_books = pd.read_parquet('df_combined_books_final.parquet')
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_model_loaded = pickle.load(f)
        
        with open('cosine_sim_matrix.pkl', 'rb') as f:
            cosine_sim_matrix_loaded = pickle.load(f)
        
        features_df = df_combined_books[['title','authors','average_rating', 'genre', 'pages']].astype(str)
        features_df['title'] = features_df['title'].str.replace(' ', '').str.lower()
        features_df['authors'] = features_df['authors'].str.replace(' ', '').str.lower()
        features_df['average_rating'] = features_df['average_rating'].str.replace(' ', '').str.lower()
        features_df['genre'] = features_df['genre'].str.replace(' ', '').str.lower()
        features_df['pages'] = features_df['pages'].str.replace(' ', '').str.lower()
        features_df['combined_features'] = features_df['title'] + ' ' + \
                                           features_df['authors'] + ' ' + \
                                           features_df['average_rating'] + ' ' + \
                                           features_df['genre'] + ' ' + \
                                           features_df['pages']

        return df_combined_books, tfidf_model_loaded, cosine_sim_matrix_loaded, features_df
    
    except FileNotFoundError as e:
        st.error(f"Error: No se encontraron los archivos de recursos necesarios. Asegúrate de que 'df_combined_books_final.parquet', 'tfidf_vectorizer.pkl' y 'cosine_sim_matrix.pkl' estén en la misma carpeta que 'content_based_app.py'.")
        st.stop()
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al cargar los recursos: {e}")
        st.stop()

df_books, tfidf_model, cosine_sim_matrix_precomputed, features_df_for_indexing = load_resources()

def recommend_books_content_based(title, df_books, tfidf_vec, cosine_sim_mtx, features_df_idx, num_recommendations=5):
    processed_title = title.lower().replace(' ', '')
    
    try:
        idx = features_df_idx[features_df_idx['title'] == processed_title].index[0]
    except IndexError:
        st.warning(f"El libro '{title}' no se encontró en nuestra base de datos para la búsqueda de similitud. Por favor, selecciona un libro de la lista.")
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim_mtx[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = []
    seen_book_ids = set()
    
    original_book_id = df_books.iloc[idx]['book_id']

    for i, score in sim_scores:
        current_book_id = df_books.iloc[i]['book_id']
        if current_book_id != original_book_id and current_book_id not in seen_book_ids:
            recommended_indices.append(i)
            seen_book_ids.add(current_book_id)
            if len(recommended_indices) >= num_recommendations:
                break
    
    if not recommended_indices:
        return pd.DataFrame()

    return df_books.iloc[recommended_indices]

# --- Configuración de Estilo (sin cambios aquí) ---
st.markdown(
    """
    <style>
    .stApp { background-color: #f7eecd; font-family: 'Roboto', 'Open Sans', 'Segoe UI', 'Arial', sans-serif; color: #333; }
    .main .block-container { padding-top: 30px; padding-bottom: 30px; }
    .stSelectbox, .stTextInput, .stSlider { font-size: 1.1em; }
    .stButton > button { background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; border: none; cursor: pointer; font-size: 1.1em; font-weight: bold; }
    .stButton > button:hover { background-color: #45a049; }
    .book-card-container { display: flex; flex-wrap: wrap; gap: 15px; padding: 15px; border-radius: 8px; background-color: #fff; border: 1px solid #ddd; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
    .book-card { flex: 0 0 auto; width: 160px; text-align: center; overflow-wrap: break-word; border: 1px solid #eee; padding: 10px; border-radius: 5px; background-color: #fcfcfc; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); font-size: 10px; display: flex; flex-direction: column; justify-content: space-between; height: 350px; }
    .book-card img { display: block; margin: 0 auto; border: 1px solid #ccc; height: 150px; width: auto; max-width: 100%; object-fit: contain; }
    .book-title { font-size: 1em; font-weight: bold; margin-bottom: 5px; height: 3em; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; color: #333; }
    .book-details { font-size: 0.85em; margin-top: 5px; color: #555; }
    .no-cover { width: 100px; height: 150px; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center; font-size: 10px; background-color: #f0f0f0; margin: 0 auto; text-align: center; padding: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Interfaz de Usuario en Streamlit ---
st.title("Book Recommender 📚")
st.markdown("### ¡Encuentra tu próxima lectura favorita!")

# --- Contenedor de filtros ---
st.sidebar.header("Filtra para encontrar un libro")

search_title = st.sidebar.text_input("Buscar por título (parcial):", "").strip()
all_genres = [''] + sorted(df_books['genre'].dropna().unique().tolist())
selected_genre = st.sidebar.selectbox("Filtrar por Género:", all_genres)
search_author = st.sidebar.text_input("Buscar por Autor (parcial):", "").strip()

min_pages_df = int(df_books['pages'].dropna().min()) if not df_books['pages'].dropna().empty else 0
max_pages_df = int(df_books['pages'].dropna().max()) if not df_books['pages'].dropna().empty else 1000
if min_pages_df > max_pages_df:
    min_pages_df = 0
    max_pages_df = 1000

page_range = st.sidebar.slider(
    "Filtrar por rango de Páginas:",
    min_value=min_pages_df,
    max_value=max_pages_df,
    value=(min_pages_df, max_pages_df)
)

filtered_df_for_selection = df_books.copy()

if search_title:
    filtered_df_for_selection = filtered_df_for_selection[
        filtered_df_for_selection['title'].str.contains(search_title, case=False, na=False)
    ]

if selected_genre and selected_genre != '':
    filtered_df_for_selection = filtered_df_for_selection[
        filtered_df_for_selection['genre'].str.lower() == selected_genre.lower()
    ]

if search_author:
    filtered_df_for_selection = filtered_df_for_selection[
        filtered_df_for_selection['authors'].str.contains(search_author, case=False, na=False)
    ]

try:
    filtered_df_for_selection['pages_numeric'] = pd.to_numeric(filtered_df_for_selection['pages'], errors='coerce')
    filtered_df_for_selection = filtered_df_for_selection[
        (filtered_df_for_selection['pages_numeric'] >= page_range[0]) & 
        (filtered_df_for_selection['pages_numeric'] <= page_range[1])
    ].drop(columns='pages_numeric')
except Exception as e:
    st.sidebar.warning(f"Error al filtrar por páginas: {e}. Asegúrate de que la columna 'pages' es numérica.")

available_book_titles = filtered_df_for_selection['title'].unique().tolist()
available_book_titles_sorted = sorted(available_book_titles)

selected_book_to_recommend = st.selectbox(
    "Selecciona un libro para encontrar recomendaciones similares:",
    options=[''] + available_book_titles_sorted
)

if selected_book_to_recommend and selected_book_to_recommend != '':
    if st.button(f"🔎 Buscar recomendaciones para '{selected_book_to_recommend}'"):
        with st.spinner("Buscando recomendaciones..."):
            recommendations = recommend_books_content_based(
                selected_book_to_recommend,
                df_books,
                tfidf_model,
                cosine_sim_matrix_precomputed,
                features_df_for_indexing,
                num_recommendations=5
            )
            
            if not recommendations.empty:
                st.markdown(f"### Libros similares a **{selected_book_to_recommend}**:")
                
                # NO ES NECESARIA LA LÍNEA 'st.markdown("<h3>¡HTML SIMPLE SE RENDERIZA!</h3>", unsafe_allow_html=True)' AQUÍ

                html_output = "<div class='book-card-container'>"
                for index, book in recommendations.iterrows():
                    # --- Preparar variables de forma más robusta ---
                    # Aseguramos que sean strings y eliminamos espacios extra
                    clean_title = str(book.get('title', "Título desconocido")).strip()
                    clean_author = str(book.get('authors', "Autor desconocido")).strip()
                    clean_genre = str(book.get('genre', "Género desconocido")).strip()
                    
                    # pages_str ya se maneja con lógica de enteros, aseguramos que sea string final
                    temp_pages = book.get('pages', "Páginas desconocidas")
                    clean_pages_str = "Páginas desconocidas"
                    try:
                        if pd.notna(temp_pages):
                            if pd.api.types.is_numeric_dtype(type(temp_pages)) and not pd.isna(temp_pages):
                                clean_pages_str = f"{int(temp_pages)} págs."
                            else:
                                clean_pages_str = str(temp_pages).strip()
                    except (ValueError, TypeError):
                        clean_pages_str = str(temp_pages).strip()


                    # Manejo de image_url: asegurar que es una URL válida y no vacía
                    raw_image_url = book.get('image_url')
                    image_tag = ""
                    if pd.notna(raw_image_url) and isinstance(raw_image_url, str) and raw_image_url.strip():
                        # Usamos .strip() para limpiar la URL también
                        clean_image_url = raw_image_url.strip()
                        # El alt del img debería ser el título del libro, escapado si fuera necesario
                        # (Streamlit suele ser indulgente, pero podemos asegurar)
                        # from html import escape # Descomentar y añadir al inicio si tienes problemas con caracteres especiales en alt
                        # alt_text = escape(clean_title) 
                        alt_text = clean_title # Mantenemos simple ya que funcionó antes
                        image_tag = f'<img src="{clean_image_url}" alt="Portada de {alt_text}">'
                    else:
                        image_tag = '<div class="no-cover">No hay<br>portada</div>'
                    # --------------------------------------------------

                    html_output += f"""
                    <div class="book-card">
                        <p class="book-title">{clean_title}</p>
                        {image_tag}
                        <div class="book-details">
                            <p>Autor: {clean_author}</p>
                            <p>Género: {clean_genre}</p>
                            <p>{clean_pages_str}</p>
                        </div>
                    </div>
                    """
                html_output += "</div>"
                
                # --- DEBUGGING TEMPORAL DE HTML (mantener esta línea por ahora) ---
                st.subheader("DEBUG: Contenido HTML de las recomendaciones (solo para inspección)")
                st.code(html_output) 
                # -----------------------------------------------------------------

                # ¡Esta es la línea clave que debe renderizar el HTML!
                st.markdown(html_output, unsafe_allow_html=True)
            else:
                st.warning(f"No se encontraron recomendaciones similares para '{selected_book_to_recommend}'.")
else:
    st.info("Utiliza los filtros de la izquierda para acotar tu búsqueda, o selecciona un libro del menú desplegable.")