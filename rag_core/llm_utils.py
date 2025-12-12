import streamlit as st
import os
from google import genai
from google.genai.errors import APIError

from google.genai import types

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

SYSTEM_INSTRUCTION_RAG = """
Vous êtes un agent d'assistance de voyage expert et un commercial très professionnel de l'agence Alpha. Votre mission est de répondre aux questions des utilisateurs en utilisant EXCLUSIVEMENT le CONTEXTE FACTUEL fourni.

Règles à suivre IMPÉRATIVEMENT :
1. **Réponse Factuelle et Courte :** Basez-vous uniquement sur les informations contenues dans le CONTEXTE. Synthétisez-les de manière claire et concise en français.
2. **Ton Professionnel :** Adoptez un ton engageant, commercial et informatif.
3. **Honnêteté :** Si le contexte ne contient PAS l'information demandée, vous devez répondre poliment : "Je suis désolé, je n'ai pas trouvé d'information précise dans nos documents de voyage concernant cette requête."
4. **Format :** Ne faites pas référence au "contexte" ou aux "documents" dans votre réponse finale.
"""

def generer_reponse_rag(client: genai.Client, question_utilisateur: str, contexte_recupere: str) -> str:
    """
    Génère la réponse finale en utilisant Gemini, en augmentant le prompt
    avec le contexte factuel récupéré par le RAG.
    
    :param client: Instance du client Gemini.
    :param question_utilisateur: La question normalisée posée par l'utilisateur.
    :param contexte_recupere: Le texte de contexte pertinent extrait du Vector Store.
    :return: La réponse synthétisée par le LLM.
    """
    
    # 1. Construction du Prompt Augmenté (le prompt principal injectant les données)
    prompt_complet = f"""
    CONTEXTE FACTUEL :
    ---
    {contexte_recupere}
    ---
    
    QUESTION DE L'UTILISATEUR :
    {question_utilisateur}
    
    Répondez à la question en utilisant le CONTEXTE FACTUEL ci-dessus et en respectant les instructions.
    """

    # 2. Configuration et Appel à l'API Gemini
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_complet,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION_RAG,
                # Basse température pour une réponse factuelle et peu créative
                temperature=0.1 
            )
        )
        return response.text
    except Exception as e:
        st.error(f"Erreur lors de la génération de la réponse finale par Gemini : {e}")
        return "Une erreur interne est survenue lors de la tentative de génération de la réponse."