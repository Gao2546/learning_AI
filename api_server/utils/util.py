import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from sentence_transformers import SentenceTransformer
import psycopg2
import dotenv
import os

dotenv.load_dotenv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_pdf_text(file_storage) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Accepts a Flask FileStorage object (e.g. from request.files).
    """
    # Load the PDF from a byte stream (not from disk)
    pdf_bytes = file_storage.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text() + "\n"

    doc.close()
    return text.strip()


def extract_image_text(file_storage) -> str:
    """
    Extract text from an image using pytesseract.
    Accepts a Flask FileStorage object (e.g. from request.files).
    """
    # Read the image bytes and convert to a PIL image
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image, lang='tha+eng')

    return text.strip()


def encode_text_for_embedding(text: str) -> list[float]:
    """
    Convert text into an embedding vector using SentenceTransformer.
    """
    embedding = model.encode(text)
    return embedding.tolist()


def save_vector_to_db(user_id, chat_history_id, file_name, text, embedding):
    """
    Save embedding to the document_embeddings table in PostgreSQL using pgvector.
    """
    # Convert Python list to PostgreSQL vector literal (e.g., '[0.1, 0.2, 0.3]')
    vector_literal = f"[{', '.join(map(str, embedding))}]"

    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PGDATABASE", "ai_agent"),
            user=os.getenv("PGUSER", "athip"),
            password=os.getenv("PGPASSWORD", "123456"),
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
        )
        cur = conn.cursor()

        query = """
            INSERT INTO document_embeddings (user_id, chat_history_id, file_name, extracted_text, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """

        cur.execute(query, (user_id, chat_history_id, file_name, text, vector_literal))

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Vector saved to database.")
    except Exception as e:
        print("❌ Failed to save vector:", e)


def search_similar_documents_by_chat(query_text: str, user_id: int, chat_history_id: int, top_k: int = 5):
    """
    Search for the top-k most similar document embeddings for a given chat_history_id and user_id.
    """
    # Step 1: Encode the query text to a vector
    query_embedding = model.encode(query_text).tolist()
    query_vector = f"[{', '.join(map(str, query_embedding))}]"

    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PGDATABASE", "ai_agent"),
            user=os.getenv("PGUSER", "athip"),
            password=os.getenv("PGPASSWORD", "123456"),
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432"),
        )
        cur = conn.cursor()

        # Step 2: Search within same user and same chat
        query = """
            SELECT 
                id, file_name, extracted_text, embedding <=> %s AS distance
            FROM document_embeddings
            WHERE user_id = %s AND chat_history_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        cur.execute(query, (query_vector, user_id, chat_history_id, query_vector, top_k))
        results = cur.fetchall()

        cur.close()
        conn.close()

        # Step 3: Return results
        return [
            {
                'id': row[0],
                'file_name': row[1],
                'text': row[2],
                'distance': row[3]
            }
            for row in results
        ]

    except Exception as e:
        print("❌ Failed to perform similarity search:", e)
        return []
