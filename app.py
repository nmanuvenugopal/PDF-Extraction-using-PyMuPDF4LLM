import os
import tempfile
from pathlib import Path
import logging

import streamlit as st
import pymupdf4llm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from llama_index.core import Document, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.schema import ImageNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.clip import ClipEmbedding
import qdrant_client
from qdrant_client.http.models import Distance, VectorParams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self):
        """Initialize the PDF Extractor with necessary directories and database connections."""
        # Create image directory if it doesn't exist
        self.img_path = Path("images")
        self.img_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client and vector stores
        self.setup_vector_stores()
        
        # Track processing state
        self.is_processed = False
        self.documents = []
        self.image_documents = []
        self.index = None

    def setup_vector_stores(self):
        """Set up Qdrant vector stores for text and image embeddings."""
        try:
            # Initialize Qdrant client (in-memory for demo purposes)
            self.client = qdrant_client.QdrantClient(location=":memory:")
            
            # Create collections for text and image embeddings
            self.client.create_collection(
                collection_name="text_collection",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            self.client.create_collection(
                collection_name="image_collection",
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            
            # Initialize vector stores
            self.text_store = QdrantVectorStore(client=self.client, collection_name="text_collection")
            self.image_store = QdrantVectorStore(client=self.client, collection_name="image_collection")
            
            logger.info("Vector stores initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up vector stores: {e}")
            st.error(f"Failed to initialize vector stores: {e}")

    def process_pdf(self, pdf_file):
        """Process the uploaded PDF file, extracting text and images."""
        if pdf_file is None:
            st.warning("Please upload a PDF file")
            return False
        
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Saving PDF file...")
            
            # Save uploaded PDF to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.read())
                temp_file_path = temp_file.name
            
            progress_bar.progress(20)
            status_text.text("Extracting content from PDF...")
            
            # Extract content and images from PDF
            docs = pymupdf4llm.to_markdown(
                doc=temp_file_path,
                page_chunks=True,
                write_images=True,
                image_path=str(self.img_path),
                image_format="jpg"
            )
            
            progress_bar.progress(50)
            status_text.text("Processing extracted content...")
            
            # Process extracted documents
            self.documents = []
            for document in docs:
                metadata = {
                    "file_path": document["metadata"].get("file_path"),
                    "page": str(document["metadata"].get("page")),
                    "images": str(document.get("images")),
                    "toc_items": str(document.get("toc_items")),
                }
                
                llama_document = Document(
                    text=document["text"],
                    metadata=metadata,
                    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
                )
                
                self.documents.append(llama_document)
            
            progress_bar.progress(70)
            status_text.text("Processing images...")
            
            # Load image documents if any images were extracted
            if os.path.exists(self.img_path) and any(os.listdir(self.img_path)):
                self.image_documents = SimpleDirectoryReader(str(self.img_path)).load_data()
            else:
                self.image_documents = []
                logger.info("No images found in the PDF")
            
            progress_bar.progress(80)
            status_text.text("Creating vector index...")
            
            # Create storage context and index
            storage_context = StorageContext.from_defaults(
                vector_store=self.text_store, 
                image_store=self.image_store
            )
            
            # Create multimodal vector store index
            self.index = MultiModalVectorStoreIndex.from_documents(
                self.documents + self.image_documents,
                storage_context=storage_context,
                embed_model=ClipEmbedding()
            )
            
            progress_bar.progress(100)
            status_text.text("PDF processed successfully!")
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            self.is_processed = True
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            st.error(f"Failed to process PDF: {e}")
            return False

    def retrieve_information(self, query, top_k=3):
        """Retrieve information based on the query."""
        if not self.is_processed:
            st.warning("Please process a PDF file first")
            return None
        
        if not query:
            st.warning("Please enter a query")
            return None
        
        try:
            # Create retriever with specified parameters
            retriever = self.index.as_retriever(
                similarity_top_k=top_k, 
                image_similarity_top_k=top_k
            )
            
            # Retrieve results
            retrieval_result = retriever.retrieve(query)
            return retrieval_result
            
        except Exception as e:
            logger.error(f"Error retrieving information: {e}")
            st.error(f"Failed to retrieve information: {e}")
            return None

    def display_results(self, results):
        """Display retrieval results in the Streamlit UI."""
        if not results:
            return
        
        st.subheader("Retrieval Results:")
        
        # Display text results
        text_results = [r for r in results if not isinstance(r.node, ImageNode)]
        if text_results:
            st.write("### Text Content")
            for i, result in enumerate(text_results):
                with st.expander(f"Text Result {i+1} (Score: {result.score:.4f})"):
                    st.write(result.node.text)
        
        # Display image results
        image_results = [r for r in results if isinstance(r.node, ImageNode)]
        if image_results:
            st.write("### Images")
            self.plot_images(image_results)

    def plot_images(self, image_results):
        """Plot retrieved images in a grid."""
        try:
            # Get image paths from results
            image_paths = [r.node.metadata["file_path"] for r in image_results 
                          if os.path.isfile(r.node.metadata["file_path"])]
            
            if not image_paths:
                st.write("No images found in results")
                return
            
            # Determine grid size based on number of images
            num_images = len(image_paths)
            cols = min(3, num_images)
            rows = (num_images + cols - 1) // cols  # Ceiling division
            
            # Create figure with appropriate size
            fig, axarr = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            
            # Handle single image case
            if num_images == 1:
                axarr = np.array([[axarr]])
            # Handle single row case
            elif rows == 1:
                axarr = axarr.reshape(1, -1)
            
            # Plot each image
            for i, img_path in enumerate(image_paths):
                row, col = i // cols, i % cols
                try:
                    image = Image.open(img_path)
                    axarr[row, col].imshow(image)
                    axarr[row, col].axis('off')
                    axarr[row, col].set_title(f"Score: {image_results[i].score:.4f}")
                except Exception as e:
                    logger.error(f"Error displaying image {img_path}: {e}")
                    axarr[row, col].text(0.5, 0.5, f"Error loading image: {e}", 
                                        ha='center', va='center')
            
            # Hide unused subplots
            for i in range(num_images, rows * cols):
                row, col = i // cols, i % cols
                axarr[row, col].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Also display individual images with their scores for better viewing
            for i, result in enumerate(image_results):
                if os.path.isfile(result.node.metadata["file_path"]):
                    st.image(
                        result.node.metadata["file_path"], 
                        caption=f"Image {i+1} (Score: {result.score:.4f})"
                    )
                
        except Exception as e:
            logger.error(f"Error plotting images: {e}")
            st.error(f"Failed to display images: {e}")


def main():
    """Main function to run the Streamlit app."""
    # Set page configuration
    st.set_page_config(
        page_title="PDF Extraction and Retrieval",
        page_icon="📄",
        layout="wide"
    )
    
    # App title and description
    st.title("📄 PDF Extraction and Retrieval")
    st.write("""
    This application allows you to extract content from PDF files and perform 
    semantic search on both text and images. Upload a PDF file, then ask questions 
    about its content.
    """)
    
    # Initialize PDF extractor
    extractor = PDFExtractor()
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Upload and Settings")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            process_button = st.button("Process PDF")
            if process_button:
                with st.spinner("Processing PDF..."):
                    success = extractor.process_pdf(uploaded_file)
                    if success:
                        st.success("PDF processed successfully!")
        
        # Settings
        st.subheader("Search Settings")
        top_k = st.slider("Number of results to return", min_value=1, max_value=10, value=3)
    
    # Main content area
    if extractor.is_processed:
        st.subheader("Ask Questions About the PDF")
        query = st.text_input("Enter your query here:")
        
        if query:
            with st.spinner("Retrieving information..."):
                results = extractor.retrieve_information(query, top_k=top_k)
                extractor.display_results(results)
    else:
        if uploaded_file is None:
            st.info("Please upload a PDF file in the sidebar to get started.")
        else:
            st.info("Click 'Process PDF' in the sidebar to extract content.")
    
    # Footer
    st.markdown("---")
    st.caption("PDF Extraction and Retrieval Tool | Using pymupdf4llm and LlamaIndex")


if __name__ == "__main__":
    main()
