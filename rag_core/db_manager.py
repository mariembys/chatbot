# rag_core/db_manager.py (Version ultra-compacte)

import os
from typing import List
import pandas as pd
import streamlit as st 

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Variables Globales ---
DATA_PATH = "data/"
VECTOR_STORE_PATH = "vectorstore/chroma_db"
CSV_FILE_NAME = "Travel details dataset.csv" 
CSV_FILE_PATH = os.path.join(DATA_PATH, CSV_FILE_NAME)
EMBEDDING_MODEL_NAME = "sentence-transformers/LaBSE" 

# --- Fonctions de Nettoyage ---

def int_to_str(value, default="Inconnu"):
    """Convertit une valeur en entier (sécurisé) et retourne une chaîne."""
    try:
        return str(int(float(value))) 
    except (ValueError, TypeError):
        return default

def clean_and_combine_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DF et crée la colonne 'trip_summary' pour le RAG."""
    df = df.fillna('') 
    
    # Nettoyage des coûts
    for col in ['Accommodation cost', 'Transportation cost']:
        df[col] = df[col].astype(str).str.replace(r'[$,USD]', '', regex=True).str.strip()
        df[col] = df[col].apply(lambda x: 'Non spécifié' if x == '' or x == '0' else x)
    
    # Création du résumé textuel vectorisable
    df['trip_summary'] = df.apply(
        lambda row: (
            f"Voyage ID {int_to_str(row['Trip ID'])}. Destination: {row['Destination']}. "
            f"Durée: {int_to_str(row['Duration (days)'], 'Inconnue')} jours. "
            f"Hébergement: {row['Accommodation type']} (Coût: {row['Accommodation cost']}). "
            f"Transport: {row['Transportation type']} (Coût: {row['Transportation cost']}). "
            f"Voyageur: {row['Traveler name']} ({int_to_str(row['Traveler age'], 'Inconnu')} ans)."
        ), 
        axis=1
    )
    
    # Filtration des lignes vides
    return df[df['Destination'] != ''].reset_index(drop=True)

def load_csv_document() -> List[Document]:
    """Charge le CSV et le convertit en LangChain Documents."""
    if not os.path.exists(CSV_FILE_PATH):
        st.error(f" ERREUR : Fichier CSV non trouvé : {CSV_FILE_PATH}")
        return []

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df_processed = clean_and_combine_data(df)
        
        documents = []
        for _, row in df_processed.iterrows():
            metadata = {
                "trip_id": row['Trip ID'],
                "destination": row['Destination'],
                "accommodation_type": row['Accommodation type'],
                "transportation_type": row['Transportation type'],
                "traveler_nationality": row['Traveler nationality']
            }
            documents.append(Document(page_content=row['trip_summary'], metadata=metadata))
            
        return documents
    
    except Exception as e:
        st.error(f"Erreur CSV : {e}")
        return []

# --- Fonctions de Vectorisation ---
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_multilingual_embeddings():
    """Charge correctement LaBSE sans erreur meta-tensor."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": "cpu",             # Évite le .to(meta)
            "trust_remote_code": True    # Au cas où le modèle en a besoin
        },
        encode_kwargs={"normalize_embeddings": True}
    )

def create_vector_store(documents: List[Document]):
    """Crée et indexe la base vectorielle ChromaDB."""
    # Pas besoin de splitter car chaque ligne est déjà un chunk de taille raisonnable
    embeddings = get_multilingual_embeddings()
    
    db = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=VECTOR_STORE_PATH
    )
    db.persist()
    st.success(f" Base vectorielle ChromaDB créée avec {db._collection.count()} vecteurs.")
    return db

def load_existing_vector_store():
    """Charge une instance ChromaDB existante."""
    if os.path.exists(VECTOR_STORE_PATH):
        return Chroma(
            persist_directory=VECTOR_STORE_PATH, 
            embedding_function=get_multilingual_embeddings()
        )
    return None

# --- Pipeline de l'Étape 3 ---

def pipeline_complet_preparation_dataset():
    """Fonction principale du pipeline de l'Étape 3."""
    st.markdown("### 🛠️ Démarrage de l'Étape 3 : Indexation")
    
    # 1. Chargement et Traitement
    with st.spinner("1/2. Chargement et structuration des données CSV..."):
        documents = load_csv_document() 
        if not documents:
            return None
        st.success(f" {len(documents)} enregistrements de voyage chargés.")
    
    # 2. Vectorisation et Indexation
    with st.spinner(f"2/2. Création et Indexation (ChromaDB) avec {EMBEDDING_MODEL_NAME}."):
        try:
            return create_vector_store(documents)
        except Exception as e:
            st.error(f" ERREUR lors de la création de la base : {e}")
            return None