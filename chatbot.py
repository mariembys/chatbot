import streamlit as st
import os
from dotenv import load_dotenv
# Charger les variables d'environnement
load_dotenv()
# Importation des utilitaires LLM (Gemini) depuis le module
from rag_core import llm_utils 
from rag_core.anomaly_detector import AnomalyDetector
# --- Les futures imports pour le RAG (db_manager) viendront ici ---
from rag_core import db_manager 

# def main():
#     """
#     Fonction principale de l'application Streamlit.
#     Gère l'interface utilisateur et le flux de traitement de la requête.
#     """
    
#     # 1. Configuration de la page
#     st.set_page_config(
#         page_title="Agent Commercial de Voyage IA - RAG",
#         page_icon="🤖",
#         layout="wide" # Utilise toute la largeur de l'écran
#     )

#     st.title("✈️ Votre Agent Commercial de Voyage IA Multilingue")
#     st.markdown("""
#     Bienvenue ! Posez votre question concernant les voyages, en **Français**, en **Anglais**, en **Arabe** ou en **Dialecte Tunisien** (Derja).
#     """)

# # --- SIDEBAR : Configuration ---
#     with st.sidebar:
#         st.header("⚙️ Configuration du Système")
        
#         st.markdown("### 📊 Étape 3 : Préparer le Dataset")
#         st.info("""
#         Chargez vos documents de voyage depuis le dossier `data/` 
#         et créez la base vectorielle ChromaDB.
        
#         ⏱️ Première exécution : peut prendre 5-10 minutes 
#         (téléchargement du modèle LaBSE ~500 MB)
#         """)
        
#         # BOUTON POUR LANCER L'ÉTAPE 3
#         if st.button("🚀 Préparer le Dataset", type="primary", use_container_width=True):
#             db_manager.pipeline_complet_preparation_dataset()




#     st.divider()
# # Afficher l'état de la base vectorielle
#     st.markdown("### 📈 État du Système")
#     if os.path.exists("vectorstore/chroma_db"):
#         st.success("✅ Base vectorielle créée")
#     else:
#         st.warning("⚠️ Base non créée")
#     st.divider()



#     # 2. Zone de Saisie de la Requête Client
#     requete_client = st.text_area(
#         "Entrez votre requête ici :",
#         height=150,
#         placeholder="Ex: نحب نسافر لتونس في الصيف. | I want to book a flight to Paris. | Je cherche des infos sur la visa pour Dubaï."
#     )

#     # 3. Bouton de Soumission et Déclenchement du Pipeline
#     if st.button("Chercher l'Information", type="primary"):
        
#         # Vérification de l'input
#         if not requete_client:
#             st.warning("Veuillez entrer une requête pour commencer.")
#             return # Arrêter l'exécution si la requête est vide
            
#         # --- ÉTAPE 1 : Préparation ---
#         st.info(f"Requête initiale : **{requete_client}**")
        
#         # Initialisation du client Gemini (la fonction vérifie la clé API)
#         gemini_client = llm_utils.get_gemini_client()
        
#         st.divider()
        
#         # --- ÉTAPE 2 : Traitement Multilingue et Normalisation ---
#         with st.spinner("⏳ Étape 2: Traduction et normalisation de la requête (Gemini)..."):
#             requete_normalisee = llm_utils.traiter_requete_multilingue(
#                 gemini_client, requete_client
#             )

#         if requete_normalisee:
#             st.success("✅ ÉTAPE 2 RÉUSSIE : Requête normalisée (en Français) :")
#             st.code(requete_normalisee, language='text')

            
            

# if __name__ == "__main__":
#     main()
# chatbot.py (Fonction main)

import streamlit as st
import os
from dotenv import load_dotenv
# Charger les variables d'environnement
load_dotenv()

# Importation des utilitaires LLM (Gemini)
from rag_core import llm_utils 
# Importation du gestionnaire de base de données (RAG Indexing)
from rag_core import db_manager 
# NOUVEAU : Importation du Détecteur d'Anomalies
from rag_core.anomaly_detector import AnomalyDetector 

def main():
    """
    Fonction principale de l'application Streamlit.
    Gère l'interface utilisateur et le flux de traitement de la requête.
    """
    
    # 1. Configuration de la page
    st.set_page_config(
        page_title="Agent Commercial de Voyage IA - RAG",
        page_icon="🤖",
        layout="wide"
    )

    st.title("✈️ Votre Agent Commercial de Voyage IA Multilingue")
    st.markdown("""
    Bienvenue ! Posez votre question concernant les voyages, en **Français**, en **Anglais**, en **Arabe** ou en **Dialecte Tunisien** (Derja).
    """)

# --- SIDEBAR : Configuration et Initialisation ---
    with st.sidebar:
        st.header("⚙️ Configuration du Système")
        
        st.markdown("### 📊 Étape 3 : Préparer le Dataset")
        st.info("""
        Chargez vos documents de voyage et créez la base vectorielle ChromaDB.
        """)
        
        # BOUTON POUR LANCER L'ÉTAPE 3
        if st.button("🚀 Préparer le Dataset", type="primary", use_container_width=True):
            vectorstore = db_manager.pipeline_complet_preparation_dataset()
            
            # Stockage de la base vectorielle et du détecteur dans l'état de session Streamlit
            if vectorstore:
                st.session_state['vectorstore'] = vectorstore
                # NOUVEAU : Initialiser le détecteur après la création de la base
                st.session_state['anomaly_detector'] = AnomalyDetector(vectorstore)


    st.divider()
    
# --- Chargement et Affichage de l'État du Système ---

    # Tenter de charger l'index RAG et le détecteur au premier chargement de la page
    if 'vectorstore' not in st.session_state and os.path.exists(db_manager.VECTOR_STORE_PATH):
        with st.spinner("⏳ Chargement de l'index de voyage existant..."):
            vectorstore = db_manager.load_existing_vector_store()
            st.session_state['vectorstore'] = vectorstore
            
            if vectorstore:
                st.session_state['anomaly_detector'] = AnomalyDetector(vectorstore) # Initialisation du détecteur
    
    # Afficher l'état de la base vectorielle
    if st.session_state.get('vectorstore'):
        st.success("✅ Base vectorielle & Détecteur d'anomalies chargés.")
    else:
        st.warning("⚠️ Base de données non créée. Veuillez cliquer sur 'Préparer le Dataset'.")

    st.divider()



    # 2. Zone de Saisie de la Requête Client
    requete_client = st.text_area(
        "Entrez votre requête ici :",
        height=150,
        placeholder="Ex: نحب نسافر لتونس في الصيف. | I want to book a flight to Paris. | Je cherche des infos sur la visa pour Dubaï."
    )

    # 3. Bouton de Soumission et Déclenchement du Pipeline
    if st.button("Chercher l'Information", type="primary"):
        
        # Vérification des prérequis RAG
        if not st.session_state.get('vectorstore'):
            st.error("Impossible de chercher : la base de données vectorielle n'est pas chargée.")
            return

        # Vérification de l'input
        if not requete_client:
            st.warning("Veuillez entrer une requête pour commencer.")
            return 
            
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
            
            st.divider()

            # --- NOUVEAU BLOC : Contrôle du Sujet (Isolation Forest) ---
            st.markdown("### 🔎 Contrôle de Pertinence du Sujet")
            detector = st.session_state['anomaly_detector']
            
            with st.spinner("⏳ Vérification du sujet de la requête (Isolation Forest)..."):
                is_outlier = detector.is_anomaly(requete_normalisee)
                
            if is_outlier:
                # Si l'anomalie est détectée, nous arrêtons le RAG et affichons le message
                st.error("🚫 SUJET HORS-CONTRÔLE : Votre requête est jugée hors sujet par le système. Veuillez vous concentrer uniquement sur les destinations, coûts, ou types de voyages présents dans notre dataset.")
                return 
            else:
                st.success("✅ Sujet pertinent détecté. Lancement du RAG.")
            # --------------------------------------------------------
            
            st.divider()
            
            # --- ÉTAPE 3 : Recherche RAG (Retrieval) ---
            st.markdown("### 🔍 ÉTAPE 3 : Recherche de Contexte")
            vectorstore = st.session_state['vectorstore']
            
            with st.spinner("⏳ Recherche de contexte pertinent dans la base de données..."):
                contexte_trouve = db_manager.search_db(
                    requete_normalisee, 
                    vectorstore,
                    k=3
                )
            
            if contexte_trouve:
                st.success("✅ Contexte(s) récupéré(s) :")
                st.code(contexte_trouve, language='markdown')

                st.divider()

                # --- ÉTAPE 4 : Génération Augmentée (Generation) ---
                st.markdown("### 💬 ÉTAPE 4 : Génération Augmentée")
                
                with st.spinner("⏳ Génération de la réponse finale avec Gemini..."):
                    reponse_finale = llm_utils.generer_reponse_rag(
                        gemini_client, 
                        requete_normalisee, 
                        contexte_trouve
                    )
                
                if reponse_finale:
                    st.success("🤖 Réponse de l'Agent IA :")
                    st.markdown(reponse_finale) 
                else:
                    st.error("La génération de la réponse finale a échoué.")
            else:
                 st.warning("⚠️ Aucun contexte pertinent trouvé. La réponse sera générale ou basée sur un contexte vide.")
                # Si aucun contexte, on pourrait fallback sur une réponse LLM pure


if __name__ == "__main__":
    main()