import streamlit as st

def main():
    # 1. Configuration de la page
    st.set_page_config(
        page_title="Agent Commercial de Voyage IA - RAG",
        page_icon="🤖"
    )

    st.title("✈️ Votre Agent Commercial de Voyage IA Multilingue")
    st.markdown("""
    Bienvenue ! Posez votre question concernant les voyages, en **Français**, en **Anglais**, en **Arabe** ou en **Dialecte Tunisien** (Derja).
    """)
    st.divider()

    # 2. Zone de Saisie de la Requête Client
    # Cette variable `requete_client` contiendra l'input de l'utilisateur
    requete_client = st.text_area(
        "Entrez votre requête ici :",
        height=150,
        placeholder="Ex: نحب نسافر لتونس في الصيف. | I want to book a flight to Paris. | Je cherche des infos sur la visa pour Dubaï."
    )

    # 3. Bouton de Soumission
    if st.button("Chercher l'Information", type="primary"):
        if requete_client:
            # L'étape suivante (2. Traitement Multilingue) se fera ici
            
            # --- ÉTAPE 2 : Début du Traitement ---
            # Nous affichons d'abord l'input pour confirmation.
            st.info(f"Requête reçue (langue inconnue) : **{requete_client}**")
            
            # Appel à la fonction de traitement PNL / RAG (à créer)
            
            # **TO DO (À venir) :** intégrer les étapes XLM-RoBERTa et Gemini ici
            
            # Placeholder pour le résultat final
            # st.success("Réponse de l'IA (en attente d'intégration du RAG) : ...")

        else:
            st.warning("Veuillez entrer une requête pour commencer.")


if __name__ == "__main__":
    main()