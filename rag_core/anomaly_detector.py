# rag_core/anomaly_detector.py

import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_community.vectorstores import Chroma

# Le modèle doit être le même que celui utilisé pour la vectorisation
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE" 

class AnomalyDetector:
    """
    Détecte les requêtes utilisateurs sémantiquement hors-sujet par rapport 
    à la base de connaissances indexée.
    """
    def __init__(self, vectorstore: Chroma):
        # 1. Récupération de TOUS les vecteurs de la DB
        self.vectorstore = vectorstore
        self.embeddings_data = self._load_all_embeddings()
        
        # 2. Initialisation et Entraînement de l'Isolation Forest
        self.model = IsolationForest(
            contamination='auto', # La contamination est la proportion d'anomalies
            random_state=42
        )
        # Entraînement sur la totalité des vecteurs de la base de voyage (le sujet principal)
        self.model.fit(self.embeddings_data)
        st.success("🌲 Modèle Isolation Forest prêt pour la détection d'anomalies.")

    def _load_all_embeddings(self) -> np.ndarray:
        """Extrait tous les vecteurs de la collection ChromaDB de manière sécurisée."""
        collection = self.vectorstore._collection

        try:
            results = collection.get(include=['embeddings'])
        except Exception as e:
            st.error(f"Erreur lors de la récupération des embeddings : {e}")
            return np.array([])

        embeddings = results.get('embeddings')

        # ✅ Vérification sécurisée
        if embeddings is None or len(embeddings) == 0:
            st.error("⚠️ Aucun embedding trouvé dans la base Chroma.")
            return np.array([])

        st.success(f"✅ {len(embeddings)} embeddings chargés dans le détecteur.")
        return np.array(embeddings)

    
    def get_embeddings_function(self):
        """Retourne la fonction d'embedding utilisée (pour vectoriser la requête)."""
        return HuggingFaceBgeEmbeddings(
             model_name=EMBEDDING_MODEL_NAME,
             model_kwargs={'device': 'cpu'} 
        )

    def is_anomaly(self, query_text: str, threshold: float = -0.5) -> bool:
        """
        Détermine si la requête utilisateur est une anomalie (hors-sujet).
        Retourne True si c'est une anomalie (hors sujet).
        """
        # 1. Vectoriser la requête
        embedding_fn = self.get_embeddings_function()
        query_vector = embedding_fn.embed_query(query_text)
        query_vector = np.array(query_vector).reshape(1, -1)
        
        # 2. Prédire le score d'anomalie
        # Le score renvoie la 'distance' du point par rapport aux données normales
        anomaly_score = self.model.decision_function(query_vector)[0]
        
        st.info(f"Score d'anomalie pour la requête : {anomaly_score:.2f} (Seuil : {threshold})")
        
        # 3. Déterminer si c'est une anomalie
        # Si le score est inférieur au seuil, c'est une anomalie (False pour inlier, True pour outlier)
        return anomaly_score < threshold