import streamlit as st
import os
from google import genai
from google.genai.errors import APIError

# --- Configuration et Initialisation Gemini ---

def get_gemini_client():
    """
    Vérifie la clé API et initialise le client Gemini.
    Arrête l'application Streamlit en cas d'erreur.
    """
    # La librairie 'google-genai' recherche la clé dans GEMINI_API_KEY
    if "GEMINI_API_KEY" not in os.environ:
        st.error("ERREUR : La variable d'environnement GEMINI_API_KEY n'est pas configurée.")
        st.stop()
    
    try:
        client = genai.Client()
        return client
    except Exception as e:
        # Gérer les erreurs d'initialisation (problème de connexion, etc.)
        st.error(f"Erreur d'initialisation du client Gemini : {e}")
        st.stop()


def traiter_requete_multilingue(client: genai.Client, requete_brute: str) -> str | None:
    """
    Utilise Gemini pour traduire et normaliser la requête de l'utilisateur
    en Français standard pour la recherche RAG.
    """
    prompt_normalisation = f"""
    Tu es un nettoyeur et traducteur de requêtes. L'utilisateur a saisi une requête qui peut être en français, anglais, arabe classique ou dialecte tunisien (Derja).
    
    TA TÂCHE :
    1. Si la requête est en dialecte tunisien ou en arabe classique, traduis-la en **Français standard**.
    2. Si elle est déjà en Français ou en Anglais, nettoie-la et **reformule-la de manière concise** pour qu'elle puisse être utilisée comme une requête de recherche factuelle.
    3. Ne génère **QUE** la requête traduite et normalisée, sans aucune explication ni salutation.
    
    REQUÊTE BRUTE : "{requete_brute}"
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_normalisation
        )
        # S'assurer qu'on retire les espaces inutiles autour
        return response.text.strip()
    except APIError as e:
        st.error(f"Erreur API lors de la normalisation (code {e.response.status_code}): {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue lors de la normalisation : {e}")
        return None

