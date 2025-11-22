import streamlit as st
import os
# Importation des utilitaires LLM (Gemini) depuis le module
from rag_core import llm_utils 

# --- Les futures imports pour le RAG (db_manager) viendront ici ---
# from rag_core import db_manager 

def main():
    """
    Fonction principale de l'application Streamlit.
    Gère l'interface utilisateur et le flux de traitement de la requête.
    """
    
    # 1. Configuration de la page
    st.set_page_config(
        page_title="Agent Commercial de Voyage IA - RAG",
        page_icon="🤖",
        layout="wide" # Utilise toute la largeur de l'écran
    )

    st.title("✈️ Votre Agent Commercial de Voyage IA Multilingue")
    st.markdown("""
    Bienvenue ! Posez votre question concernant les voyages, en **Français**, en **Anglais**, en **Arabe** ou en **Dialecte Tunisien** (Derja).
    """)
    st.divider()

    # 2. Zone de Saisie de la Requête Client
    requete_client = st.text_area(
        "Entrez votre requête ici :",
        height=150,
        placeholder="Ex: نحب نسافر لتونس في الصيف. | I want to book a flight to Paris. | Je cherche des infos sur la visa pour Dubaï."
    )

    # 3. Bouton de Soumission et Déclenchement du Pipeline
    if st.button("Chercher l'Information", type="primary"):
        
        # Vérification de l'input
        if not requete_client:
            st.warning("Veuillez entrer une requête pour commencer.")
            return # Arrêter l'exécution si la requête est vide
            
        # --- ÉTAPE 1 : Préparation ---
        st.info(f"Requête initiale : **{requete_client}**")
        
        # Initialisation du client Gemini (la fonction vérifie la clé API)
        gemini_client = llm_utils.get_gemini_client()
        
        st.divider()
        
        # --- ÉTAPE 2 : Traitement Multilingue et Normalisation ---
        with st.spinner("⏳ Étape 2: Traduction et normalisation de la requête (Gemini)..."):
            requete_normalisee = llm_utils.traiter_requete_multilingue(
                gemini_client, requete_client
            )

        if requete_normalisee:
            st.success("✅ ÉTAPE 2 RÉUSSIE : Requête normalisée (en Français) :")
            st.code(requete_normalisee, language='text')

            
            

if __name__ == "__main__":
    main()