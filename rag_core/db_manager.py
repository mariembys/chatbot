import os
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

# --- Configuration ---
DATA_DIR = "data"                          # Dossier contenant vos documents
VECTORSTORE_DIR = "vectorstore/chroma_db"  # Où stocker la base vectorielle
CHUNK_SIZE = 500                           # Taille des chunks en tokens (environ)
CHUNK_OVERLAP = 50                         # Chevauchement entre chunks


def charger_documents(data_dir: str = DATA_DIR) -> List:
    """
    Charge tous les documents texte du dossier data/
    
    Returns:
        List: Liste de documents LangChain
    """
    try:
        # Charger tous les fichiers .txt du dossier
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.txt",           # Recherche récursive des .txt
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        
        st.success(f"✅ {len(documents)} document(s) chargé(s) depuis '{data_dir}'")
        return documents
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des documents : {e}")
        return []


def decouper_documents(documents: List, chunk_size: int = CHUNK_SIZE, 
                       chunk_overlap: int = CHUNK_OVERLAP) -> List:
    """
    Découpe les documents en chunks de taille optimale
    
    Args:
        documents: Liste de documents LangChain
        chunk_size: Taille approximative de chaque chunk (en caractères)
        chunk_overlap: Nombre de caractères de chevauchement entre chunks
    
    Returns:
        List: Liste de chunks (documents plus petits)
    """
    try:
        # Initialiser le text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Priorité de découpe
        )
        
        # Découper tous les documents
        chunks = text_splitter.split_documents(documents)
        
        st.success(f"✅ {len(chunks)} chunks créés (taille: {chunk_size} caractères, overlap: {chunk_overlap})")
        return chunks
    
    except Exception as e:
        st.error(f"❌ Erreur lors du découpage : {e}")
        return []


def creer_embeddings_model():
    """
    Crée le modèle d'embeddings multilingue
    
    Utilise le modèle LaBSE (Language-agnostic BERT Sentence Embedding)
    qui supporte plus de 100 langues dont le français, l'arabe et l'anglais.
    
    Alternative : 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    
    Returns:
        HuggingFaceEmbeddings: Modèle d'embeddings
    """
    try:
        # Modèle multilingue recommandé pour votre cas d'usage
        model_name = "sentence-transformers/LaBSE"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Utilisez 'cuda' si vous avez un GPU
            encode_kwargs={'normalize_embeddings': True}  # Normalisation pour la similarité cosine
        )
        
        st.success(f"✅ Modèle d'embeddings '{model_name}' initialisé")
        return embeddings
    
    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation du modèle d'embeddings : {e}")
        return None


def creer_vectorstore(chunks: List, embeddings, vectorstore_dir: str = VECTORSTORE_DIR):
    """
    Crée ou charge la base de données vectorielle ChromaDB
    
    Args:
        chunks: Liste de chunks à vectoriser
        embeddings: Modèle d'embeddings
        vectorstore_dir: Chemin où sauvegarder la base vectorielle
    
    Returns:
        Chroma: Base de données vectorielle
    """
    try:
        # Créer le dossier si nécessaire
        os.makedirs(vectorstore_dir, exist_ok=True)
        
        # Créer la vectorstore avec ChromaDB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vectorstore_dir
        )
        
        # Sauvegarder sur disque
        vectorstore.persist()
        
        st.success(f"✅ Base vectorielle créée avec {len(chunks)} chunks dans '{vectorstore_dir}'")
        return vectorstore
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la création de la vectorstore : {e}")
        return None


def charger_vectorstore(embeddings, vectorstore_dir: str = VECTORSTORE_DIR):
    """
    Charge une base vectorielle existante
    
    Args:
        embeddings: Modèle d'embeddings (doit être le même que celui utilisé pour créer la base)
        vectorstore_dir: Chemin de la base vectorielle
    
    Returns:
        Chroma: Base de données vectorielle chargée
    """
    try:
        if not os.path.exists(vectorstore_dir):
            st.warning(f"⚠️ La base vectorielle n'existe pas dans '{vectorstore_dir}'")
            return None
        
        vectorstore = Chroma(
            persist_directory=vectorstore_dir,
            embedding_function=embeddings
        )
        
        st.success(f"✅ Base vectorielle chargée depuis '{vectorstore_dir}'")
        return vectorstore
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de la vectorstore : {e}")
        return None


def pipeline_complet_preparation_dataset():
    """
    Pipeline complet pour l'étape 3 : de vos documents à la base vectorielle
    
    Ce pipeline :
    1. Charge les documents depuis data/
    2. Les découpe en chunks
    3. Crée le modèle d'embeddings
    4. Vectorise et stocke dans ChromaDB
    
    Returns:
        Chroma: Base vectorielle prête à être utilisée
    """
    st.header("📊 Étape 3 : Préparation du Dataset Voyage")
    
    with st.spinner("📁 Chargement des documents..."):
        documents = charger_documents()
    
    if not documents:
        st.error("Aucun document trouvé. Ajoutez des fichiers .txt dans le dossier 'data/'")
        return None
    
    # Afficher un aperçu
    with st.expander("📄 Aperçu des documents chargés"):
        for i, doc in enumerate(documents[:3]):  # Afficher les 3 premiers
            st.markdown(f"**Document {i+1}** : `{doc.metadata.get('source', 'N/A')}`")
            st.text(doc.page_content[:200] + "...")
    
    st.divider()
    
    with st.spinner("✂️ Découpage des documents en chunks..."):
        chunks = decouper_documents(documents)
    
    if not chunks:
        return None
    
    # Afficher des statistiques sur les chunks
    with st.expander("📊 Statistiques des chunks"):
        st.write(f"- **Nombre total de chunks** : {len(chunks)}")
        st.write(f"- **Taille moyenne** : {sum(len(c.page_content) for c in chunks) // len(chunks)} caractères")
        st.write(f"- **Plus petit chunk** : {min(len(c.page_content) for c in chunks)} caractères")
        st.write(f"- **Plus grand chunk** : {max(len(c.page_content) for c in chunks)} caractères")
        
        # Afficher 2 exemples de chunks
        st.markdown("**Exemples de chunks :**")
        for i in range(min(2, len(chunks))):
            st.code(chunks[i].page_content, language='text')
    
    st.divider()
    
    with st.spinner("🧠 Initialisation du modèle d'embeddings multilingue..."):
        embeddings = creer_embeddings_model()
    
    if not embeddings:
        return None
    
    st.divider()
    
    with st.spinner("🔢 Vectorisation et création de la base ChromaDB (peut prendre quelques minutes)..."):
        vectorstore = creer_vectorstore(chunks, embeddings)
    
    if vectorstore:
        st.success("🎉 **ÉTAPE 3 TERMINÉE** : Base vectorielle prête !")
        st.info("💡 Vous pouvez maintenant passer à l'étape 4 : Recherche dans la base vectorielle")
    
    return vectorstore


# --- Fonction utilitaire pour afficher des infos sur la vectorstore ---
def afficher_info_vectorstore(vectorstore):
    """
    Affiche des informations sur la base vectorielle
    """
    if vectorstore:
        st.write("### 📊 Informations sur la base vectorielle")
        
        # Récupérer tous les documents
        collection = vectorstore._collection
        st.write(f"- **Nombre de vecteurs** : {collection.count()}")
        st.write(f"- **Dossier de stockage** : `{vectorstore._persist_directory}`")