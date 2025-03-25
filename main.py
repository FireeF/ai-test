import os
import json
from pathlib import Path
#core
import streamlit as st
import pinecone as pi
#llama
from llama_index.core import Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.readers.web import UnstructuredURLLoader
from llama_index.embeddings.voyageai import VoyageEmbedding
from Firecrawler import FireCrawlWebReader

# Set page config
st.set_page_config(page_title="Universal Content Scraper", layout="wide")
st.title("Universal Content Scraper")

def split_text_into_chunks(text, max_chars=4000):
    """Split text into chunks of approximately max_chars while keeping sentences intact."""
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > max_chars and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def clean_scraped_text(text):
    """Remove common footer content and privacy policy text."""
    # First, try to find and remove the entire privacy settings block
    import re
    
    # Pattern to match the entire privacy settings section
    privacy_pattern = r"Privacy settings\s*Decide which cookies.*?(?:Change settings Read more Accept|Save)"
    cleaned_text = re.sub(privacy_pattern, "", text, flags=re.DOTALL)
    
    # Additional patterns to clean up
    patterns = [
        r"FunctionalityAnalyticsAdvertising",
        r"(?:Essential|Functionality|Analytics|Advertising):.*?(?:\n|$)",
        r"This page will(?: not)? be:.*?(?=This page|Save|\n\n|$)",
        r"a contact forms, newsletter and other forms across all pages",
        r"and interaction taken",
        r"and region based on your IP number",
        r"on each page",
        r"of the statistics functions",
        r"and advertising to your interests.*?targeting cookies\.\)",
        r"we sometimes place small data files called cookies.*?websites do this too\.",
        r"Change settings Read more Accept",
        r"Save",
        # Remove any remaining cookie-related text
        r"Cookies To make this site work properly.*?(?:\n|$)",
        # Remove multiple consecutive newlines and whitespace
        r"\n\s*\n\s*\n+",
    ]
    
    # Apply all patterns
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.MULTILINE)
    
    # Final cleanup of whitespace and newlines
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'^\s+|\s+$', '', cleaned_text, flags=re.MULTILINE)
    
    return cleaned_text.strip()

def process_document(doc):
    """Process a document by cleaning and splitting if necessary."""
    cleaned_text = clean_scraped_text(doc.text)
    text_chunks = split_text_into_chunks(cleaned_text)
    
    processed_docs = []
    total_chunks = len(text_chunks)
    
    for i, chunk in enumerate(text_chunks, 1):
        # Create new metadata with chunk information
        chunk_metadata = doc.metadata.copy() if doc.metadata else {}
        chunk_metadata.update({
            'chunk_number': i,
            'total_chunks': total_chunks,
            'is_chunked': total_chunks > 1
        })
        
        # Create new document with chunk
        chunk_doc = Document(
            text=chunk,
            metadata=chunk_metadata,
            id_=f"{doc.id_}_chunk_{i}" if doc.id_ else None
        )
        processed_docs.append(chunk_doc)
    
    return processed_docs

# Initialize session state for documents
if 'documents' not in st.session_state:
    st.session_state.documents = None

# Add Pinecone API key input
pinecone_api_key = st.text_input(
    "Enter your Pinecone API Key",
    type="password",  # This will hide the API key
    help="Enter your Pinecone API key to connect to your Pinecone instance"
)

# Initialize FireCrawl reader
@st.cache_resource
def init_firecrawl():
    return FireCrawlWebReader(
        api_key=st.secrets["firecrawl_api_key"],
        mode="scrape"
    )

firecrawl_reader = init_firecrawl()

# URL input section
st.subheader("Enter URLs to Scrape")
url_input = st.text_area("Enter URLs (one per line)", height=100)
urls_to_scrape = [url.strip() for url in url_input.split("\n") if url.strip()]

# Separate buttons for scraping and indexing
col1, col2 = st.columns(2)

with col1:
    if st.button("Scrape Content"):
        if urls_to_scrape:
            with st.spinner("Scraping content from URLs..."):
                try:
                    # Use FireCrawl to scrape the URLs
                    documents = firecrawl_reader.load_data(urls=urls_to_scrape)
                    
                    # Process documents (clean and chunk if necessary)
                    processed_documents = []
                    for doc in documents:
                        processed_docs = process_document(doc)
                        processed_documents.extend(processed_docs)
                    
                    st.session_state.documents = processed_documents
                    
                    # Display results
                    st.subheader("Scraped Content")
                    current_url = None
                    for doc in st.session_state.documents:
                        url = doc.metadata.get('url', 'Unknown')
                        if url != current_url:
                            current_url = url
                            st.markdown(f"### Content from: {url}")
                        
                        chunk_info = ""
                        if doc.metadata.get('is_chunked', False):
                            chunk_info = f" (Part {doc.metadata['chunk_number']}/{doc.metadata['total_chunks']})"
                        
                        with st.expander(f"Content{chunk_info}"):
                            st.markdown("**Timestamp:** " + doc.metadata.get('timestamp', 'N/A'))
                            if doc.metadata.get('is_chunked', False):
                                st.markdown(f"**Chunk:** {doc.metadata['chunk_number']}/{doc.metadata['total_chunks']}")
                            st.markdown("**Content:**")
                            st.markdown(doc.text)
                    
                    st.success("Content successfully scraped!")
                        
                except Exception as e:
                    st.error(f"An error occurred while scraping: {str(e)}")
        else:
            st.warning("Please select URLs to scrape.")

with col2:
    # Add index name input above the Index button
    ime_indeksa = st.text_input(
        "Enter Pinecone Index Name",
        value="",  # Remove default value
        placeholder="e.g., test-index",  # Add placeholder text
        help="Enter the name of the Pinecone index where you want to store the data. The content will be indexed in the exact index name you provide.",
        key="index_name_input"
    )
    
    # Show which index will be used
    if ime_indeksa:
        st.info(f"Content will be indexed in: **{ime_indeksa}**")
    
    if st.button("Index in Pinecone"):
        if st.session_state.documents:
            if not ime_indeksa:
                st.error("Please enter an index name above.")
                st.stop()
            
            if not pinecone_api_key:
                st.error("Please enter your Pinecone API key above.")
                st.stop()
                
            with st.spinner("Storing content in Pinecone..."):
                try:
                    # Initialize Pinecone and embedding model
                    embed_model = VoyageEmbedding(
                        voyage_api_key=st.secrets["voyage_api_key"],
                        model_name="voyage-3-large",
                    )

                    pc = pi.Pinecone(api_key=pinecone_api_key)
                    
                    pinecone_index = pc.Index(ime_indeksa)

                    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace='info')
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    
                    index = VectorStoreIndex.from_documents(
                        st.session_state.documents,
                        storage_context=storage_context,
                        embed_model=embed_model
                    )
                    st.success("Content successfully indexed in Pinecone!")
                except Exception as e:
                    st.error(f"An error occurred while indexing: {str(e)}")
        else:
            st.warning("Please scrape content first before indexing.")

# Remove the index name input from sidebar
with st.sidebar:
    st.markdown("""
    ### About
    This tool scrapes content from Universal related websites and optionally stores it in a vector database.
    
    ### Process
    1. Provide Pinecone API key
    2. Provide URLs to scrape
    3. Click "Scrape Content" to fetch the content
    4. Review the scraped content
    5. Click "Index in Pinecone" to store in the database
    """)

#TEMPLATE DOKUMENTA
#Document(id_='', embedding=None, metadata={'source': ''}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
docs = [
    #    Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Standards\nIPMA's vision is "Promoting competence throughout society to enable a world in which all projects succeed." Therefore, IPMA has defined worldwide standards for competences in the domains of Project, Programme and Portfolio Management.\nFor Individuals, we have defined the Individual Competence Baseline®, IPMA ICB®, version 4.0 (for free download click here). link: http://products.ipma.world/ipma-product/icb/\nBased on the ICB4, we have also defined a competence baseline for coaches, trainers and consultants in the field of projects, programs and portfolio management, the ICB4CCT (for free download click here) link: https://shop.ipma.world/shop/ipma-standards/books-ipma-standards/individual-competence-baseline-for-consultants-coaches-and-trainers/?v=9b7d173b068d, the IPMA Reference Guide ICB4 in an Agile World (for free download click here) link: https://shop.ipma.world/shop/ipma-standards/books-ipma-standards/ipma-reference-guide-icb4-in-an-agile-world/?v=9b7d173b068d, and the IPMA Reference Guide ICB4 for PMO (for free download click here). link: https://shop.ipma.world/shop/ipma-standards/e-books-ipma-standards/ipma-reference-guide-icb4-for-pmo-ebook/?v=11aedd0e4327\nWe have defined the standard for the excellence in project management – the Project Excellence Baseline® (IPMA PEB) (for free download click here). link: http://products.ipma.world/ipma-product/peb/\nFor organisations, we have defined the Organisational Competence Baseline, IPMA OCB® (for free download click here). link: http://products.ipma.world/ipma-product/ocb/\nThe IPMA Research Evaluation Baseline® (IPMA REB) is a new IPMA standard for research in project management (for free download click here). link: https://shop.ipma.world/shop/ipma-standards/e-books-ipma-standards/research-evaluation/?v=11aedd0e4327\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
    #Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Individual Competence Baseline® – IPMA ICB4®\nThe IPMA ICB4® is the international standard on competence for project, programme and portfolio managers. The competence needed for each of these domains is defined in the following competence areas: "People" (how do you interact with the people around you, and yourself); the "Practice" of our work (needed for Projects, Programmes and Portfolios); the "Perspective" of the intiatives you're running (the context within which the initiative is run and the link to what needs to be achieved).\nRead more: https://ipma.world/ipma-standards-development-programme/icb4/\nDownload PDF file here: https://ipma.world/app/uploads/2023/01/IPMA_Main_Brochure_2017_ENG_screen.pdf\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
    #Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Project Excellence Baseline® – IPMA PEB®\nWhat defines whether a project or programme is "good" or "bad"? When we're striving for excellence in the execution of our projects and programmes, what do we mean? IPMA has given its extended definition in the PEB, the Project Excellence Baseline. This model is based on the well-known EFQM model, but is adapted to the field of project- and programme management and after over 10 years of use has been adapted to the model we're using now.\nRead more: https://ipma.world/ipma-standards-development-programme/peb/\nDownload PDF file here: https://ipma.world/app/uploads/2023/01/IPMA_Main_Brochure_2017_ENG_screen.pdf\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
    #Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Organisational Competence Baseline® – IPMA OCB®\nProject, programme and portfolio management is a key competence for many organisations. In order to acknowledge and improve this competence, you want to know where you stand, and what the opportunities are. What is your organisation's competence class? Getting to the highest level should not be the goal for organisation but getting to the right level is! What is the right level for your organisation? In the IPMA Organisational Competence Baseline®, or IPMA OCB®, we describe the competences. Now you can assess the organisational competence of your organisation.\nRead more: https://ipma.world/ipma-standards-development-programme/ocb/\nDownload PDF file here: https://ipma.world/app/uploads/2023/01/IPMA_DELTA_leaflet_pages.pdf\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
    #Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Individual Competence Baseline for Consultants, Coaches and Trainers – IPMA ICB4CCT®\nGood Project Managers do not necessarily make good Project Management Trainers, Coaches or Consultants. They are not only competent in their own domain, they need additional competences. Therefore, IPMA has developed a standard for Coaches, Consultants and Trainers, called the IPMA ICB4CCT®. This standard will be available in 2018. The internal structure of the ICB4CCT® is the same as the structure of the ICB4®. In addition to the competence elements from IPMA ICB4®, for the domains of consultancy, coaching and training IPMA ICB4CCT® defines additional elements in the competence areas Perspective, People and Practice. Some elements are common for all three domains, some are specific per domain.\nRead more: https://shop.ipma.world/shop/ipma-standards/books-ipma-standards/individual-competence-baseline-for-consultants-coaches-and-trainers/?v=796834e7a283\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
    #Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Reference Guide ICB4® in an Agile World\nThe Agile Leader is more of a phenomenon than a role. As a responsible decision-maker, how do you know that you are working with a good Agile Leader? How does someone demonstrate good leadership? Learn about the agile competences in our IPMA reference Guide ICB4® in an Agile World\nRead more: https://shop.ipma.world/shop/ipma-standards/books-ipma-standards/ipma-reference-guide-icb4-in-an-agile-world/?v=796834e7a283\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
    #Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Reference Guide ICB4® for PMO\nIPMA introduced a new standard – ICB Reference Guide for PMO in 2023. This standard is dedicated to the structures that support projects, programmes and portfolios to run them effectively and efficiently. These structures are called PMO – Project Management Offices. A PMO is defined as an organisational unit responsible for the administrative and specialists' support of the responsible management in their management of a (set of) project(s), programme(s) or portfolio(s). PMO plays a very important role in designing, performing, monitoring and reporting activities. Specialists are working in PMO together with the Head of PMO who is leading the unit towards its goals and objectives. The new standard defines the competences for the individuals working in PMO. All the competences are aligned with the IPMA ICB that is used by all the project, programme or portfolio managers in their everyday activities.\nRead more: https://shop.ipma.world/shop/ipma-standards/e-books-ipma-standards/ipma-reference-guide-icb4-for-pmo-ebook/?v=8cee5050eeb7\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),
    #Document(id_='https://ipma.world/ipma-standards-development-programme/', embedding=None, metadata={'source': 'https://ipma.world/ipma-standards-development-programme/'}, excluded_embed_metadata_keys=[], excluded_llm_metadata_keys=[], relationships={}, text='IPMA Research Evaluation Baseline – REB®\nThe IPMA Research Evaluation Baseline® helps the various users in the field of Project, Portfolio and Program Management to guide the possibilities and results in research and is a systematic and transparent basis for the IPMA Research Awards, Best Paper Prizes and the IPMA Research conferences. The main purpose of the IPMA Research Evaluation Model (REM) is to guide different user groups to evaluate the capability and achievements of their research conducted in the domain of PPP management. The IPMA REB® can be used by these organizations and/or individuals for developing their internal research evaluation system, such as research funding agencies, research performing organizations, research customers, researchers (scientists), research evaluators, research managers, students/supervisors, and so on.\nRead more: https://shop.ipma.world/shop/ipma-standards/books-ipma-standards/research-evaluation-baseline/?v=8cee5050eeb7\n', start_char_idx=None, end_char_idx=None, text_template='{metadata_str}\n\n{content}', metadata_template='{key}: {value}', metadata_seperator='\n'),

    ]

import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

      
    


import datetime
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.write("Current date and time:", current_time)




# Embed the documents
#os.environ["OPENAI_API_KEY"] = st.secrets.openai_api_key
#embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)

