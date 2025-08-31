import os
import re
import time
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from docx import Document as DocxDocument
import psycopg
from psycopg.rows import namedtuple_row
from sentence_transformers import SentenceTransformer
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Load environment variables
load_dotenv()

# Configuration
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "postgresql://vector_user:secure_password123@localhost:5432/vector_db")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "contracts")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("contract_embedder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("contract_embedder")

# Initialize Flask app
app = Flask(__name__)

class ContractSectionClauseExtractor:
    """Extracts sections and clauses from contracts, preserving their hierarchy and integrity."""

    def __init__(self):
        # Patterns for sections and clauses
        self.section_pattern = re.compile(
            r'(?P<section_heading>(Section|SECTION|Article|ARTICLE|CHAPTER|PART)\s+[IVXLCDM0-9]+\.?.*)', re.IGNORECASE)
        self.clause_pattern = re.compile(
            r'^\s*([0-9]+(\.[0-9]+)*[\.\)]?)\s+(.+)', re.MULTILINE)
        self.table_pattern = re.compile(r'^\s*(\+[-+]+\+|\|.*\|)\s*$', re.MULTILINE)

    def extract_sections_and_clauses(self, text: str, source: str) -> List[Dict]:
        """Extract sections and their clauses from contract text."""
        sections = []
        section_blocks = self.section_pattern.split(text)
        # section_blocks alternates: [before, heading1, after1, heading2, after2, ...]
        if len(section_blocks) < 3:
            # No sections found, treat whole text as one section
            return self.extract_clauses_as_section(text, "FULL_DOCUMENT", source)

        for i in range(1, len(section_blocks), 2):
            heading = section_blocks[i].strip()
            section_text = section_blocks[i + 1]
            clauses = self.extract_clauses_as_section(section_text, heading, source)
            sections.extend(clauses)
        return sections

    def extract_clauses_as_section(self, section_text: str, section_heading: str, source: str) -> List[Dict]:
        """Extract clauses from a section block, keeping them intact and including tables."""
        clauses = []
        matches = list(self.clause_pattern.finditer(section_text))
        if not matches:
            # Fallback: treat the whole section as one clause
            cleaned = self._extract_tables_and_clean(section_text)
            if cleaned:
                clauses.append({
                    'type': 'clause',
                    'section': section_heading,
                    'clause_number': None,
                    'text': cleaned,
                    'source': source,
                    'metadata': {
                        'extraction_method': 'section_fallback',
                        'section_heading': section_heading
                    }
                })
            return clauses

        for idx, match in enumerate(matches):
            clause_num = match.group(1)
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section_text)
            clause_text = section_text[start:end].strip()
            clause_text = self._extract_tables_and_clean(clause_text)
            if len(clause_text) > 30:
                clauses.append({
                    'type': 'clause',
                    'section': section_heading,
                    'clause_number': clause_num,
                    'text': clause_text,
                    'source': source,
                    'metadata': {
                        'extraction_method': 'section_clause_pattern',
                        'section_heading': section_heading,
                        'clause_number': clause_num
                    }
                })
        return clauses

    def _extract_tables_and_clean(self, text: str) -> str:
        """Detect tables and keep them as part of the clause text, cleaning up whitespace."""
        lines = text.split('\n')
        cleaned_lines = []
        in_table = False
        table_lines = []
        for line in lines:
            if self.table_pattern.match(line):
                in_table = True
                table_lines.append(line)
            else:
                if in_table and table_lines:
                    cleaned_lines.append('\n'.join(table_lines))
                    table_lines = []
                    in_table = False
                cleaned_lines.append(line)
        if in_table and table_lines:
            cleaned_lines.append('\n'.join(table_lines))
        # Clean up whitespace, but preserve table formatting
        return '\n'.join([re.sub(r'\s+', ' ', l).strip() if not self.table_pattern.match(l) else l for l in cleaned_lines if l.strip()])

    def extract_from_pdf(self, filepath: str) -> List[Dict]:
        text = ""
        try:
            doc = fitz.open(filepath)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            filename = os.path.basename(filepath)
            return self.extract_sections_and_clauses(text, filename)
        except Exception as e:
            logger.error(f"Error extracting from PDF {filepath}: {e}")
            return []

    def extract_from_docx(self, filepath: str) -> List[Dict]:
        text = ""
        try:
            doc = DocxDocument(filepath)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            filename = os.path.basename(filepath)
            return self.extract_sections_and_clauses(text, filename)
        except Exception as e:
            logger.error(f"Error extracting from DOCX {filepath}: {e}")
            return []

    def extract_from_text_file(self, filepath: str) -> List[Dict]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            filename = os.path.basename(filepath)
            return self.extract_sections_and_clauses(text, filename)
        except Exception as e:
            logger.error(f"Error extracting from text file {filepath}: {e}")
            return []

    def process_contract(self, filepath: str) -> List[Dict]:
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.pdf':
            return self.extract_from_pdf(filepath)
        elif file_ext == '.docx':
            return self.extract_from_docx(filepath)
        elif file_ext in ['.txt', '.rtf']:
            return self.extract_from_text_file(filepath)
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return []

class ContractEmbeddingEngine:
    def __init__(self, connection_string: str = DB_CONNECTION_STRING):
        self.connection_string = connection_string
        self.model = None
        self.conn = None
        self.section_clause_extractor = ContractSectionClauseExtractor()
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg.connect(self.connection_string)
            register_vector(self.conn)
            logger.info("✅ Connected to database successfully")
            return True
        except psycopg.OperationalError as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("✅ Database connection closed")
    
    def setup_database(self) -> bool:
        """Initialize database with pgvector extension and clauses table (with section info)"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("""
                CREATE TABLE IF NOT EXISTS contract_clauses (
                    id SERIAL PRIMARY KEY,
                    clause_text TEXT NOT NULL,
                    clause_number TEXT,
                    section_heading TEXT,
                    source_document TEXT NOT NULL,
                    metadata JSONB,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS clauses_embedding_idx 
                ON contract_clauses 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS clauses_metadata_idx 
                ON contract_clauses 
                USING GIN (metadata)
                """)
                cur.execute("""
                CREATE INDEX IF NOT EXISTS clauses_source_idx 
                ON contract_clauses (source_document)
                """)
            self.conn.commit()
            logger.info("✅ Database initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            self.conn.rollback()
            return False
    
    def load_model(self):
        """Load the embedding model"""
        try:
            logger.info("🧠 Loading embedding model...")
            start = time.time()
            self.model = SentenceTransformer(MODEL_NAME)
            logger.info(f"✅ Model loaded in {time.time() - start:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def process_contract_file(self, filepath: str) -> int:
        """Process a single contract file and extract sectioned clauses"""
        try:
            filename = os.path.basename(filepath)
            clauses = self.section_clause_extractor.process_contract(filepath)
            processed_count = 0
            with self.conn.transaction():
                with self.conn.cursor() as cur:
                    for clause in clauses:
                        embedding = self.model.encode(clause['text']).tolist()
                        cur.execute(
                            """INSERT INTO contract_clauses 
                            (clause_text, clause_number, section_heading, source_document, metadata, embedding) 
                            VALUES (%s, %s, %s, %s, %s, %s)""",
                            (clause['text'], clause.get('clause_number'), clause.get('section'), clause['source'], 
                             json.dumps(clause['metadata']), embedding)
                        )
                        processed_count += 1
            logger.info(f"✅ Processed {processed_count} clauses from {filename}")
            return processed_count
        except Exception as e:
            logger.error(f"❌ Failed to process {filepath}: {e}")
            return 0
    
    def process_all_contracts(self) -> int:
        """Process all contracts in the specified directory"""
        total_processed = 0
        
        if not os.path.exists(DOCUMENTS_DIR):
            os.makedirs(DOCUMENTS_DIR)
            logger.info(f"📁 Created contracts directory: {DOCUMENTS_DIR}")
            return 0
        
        supported_extensions = ['.pdf', '.docx', '.txt']
        
        for filename in os.listdir(DOCUMENTS_DIR):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            if os.path.isfile(filepath) and any(filename.endswith(ext) for ext in supported_extensions):
                total_processed += self.process_contract_file(filepath)
        
        logger.info(f"✅ Processed {total_processed} clauses total")
        return total_processed
    
    def semantic_search(self, query: str, top_k: int = 5, 
                       source_filter: str = None) -> List[Dict]:
        """Find most similar contract clauses for a given query"""
        try:
            query_embedding = self.model.encode(query).tolist()
            sql = """
            SELECT id, clause_text, clause_number, section_heading, source_document, metadata, 
                   created_at, embedding <=> %s AS distance
            FROM contract_clauses
            """
            params = [query_embedding]
            if source_filter:
                sql += " WHERE source_document = %s"
                params.append(source_filter)
            sql += " ORDER BY distance LIMIT %s"
            params.append(top_k)
            with self.conn.cursor(row_factory=namedtuple_row) as cur:
                cur.execute(sql, params)
                results = []
                for row in cur.fetchall():
                    results.append({
                        'id': row.id,
                        'clause_text': row.clause_text,
                        'clause_number': row.clause_number,
                        'section_heading': row.section_heading,
                        'source_document': row.source_document,
                        'metadata': row.metadata,
                        'created_at': row.created_at.isoformat(),
                        'similarity': 1 - row.distance
                    })
                return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    def get_clause_count(self) -> int:
        """Get the total number of clauses in the database"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM contract_clauses")
                return cur.fetchone()[0]
        except Exception as e:
            logger.error(f"❌ Failed to get clause count: {e}")
            return 0
    
    def get_contracts_list(self) -> List[str]:
        """Get list of all processed contracts"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT DISTINCT source_document FROM contract_clauses")
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"❌ Failed to get contracts list: {e}")
            return []

# Initialize the embedding engine
embedding_engine = ContractEmbeddingEngine()

# Flask Routes
@app.route('/')
def index():
    return jsonify({
        "message": "Contract Clause Embedding API",
        "status": "running",
        "version": "1.0.0"
    })

@app.route('/setup', methods=['POST'])
def setup_database():
    """Initialize the database and load models"""
    try:
        if not embedding_engine.connect():
            return jsonify({"error": "Database connection failed"}), 500
        
        if not embedding_engine.setup_database():
            return jsonify({"error": "Database setup failed"}), 500
        
        if not embedding_engine.load_model():
            return jsonify({"error": "Model loading failed"}), 500
        
        return jsonify({"message": "Setup completed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/contracts/process', methods=['POST'])
def process_contracts():
    """Process all contracts in the contracts directory"""
    try:
        count = embedding_engine.process_all_contracts()
        return jsonify({
            "message": f"Processed {count} contract clauses",
            "count": count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['GET'])
def search():
    """Search for similar contract clauses"""
    try:
        query = request.args.get('q', '')
        top_k = int(request.args.get('top_k', 5))
        source_filter = request.args.get('source')
        
        if not query:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        
        results = embedding_engine.semantic_search(query, top_k, source_filter)
        
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get system status"""
    try:
        count = embedding_engine.get_clause_count()
        contracts = embedding_engine.get_contracts_list()
        return jsonify({
            "clause_count": count,
            "contracts_processed": contracts,
            "status": "ready" if count > 0 else "no_contracts"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/contracts/upload', methods=['POST'])
def upload_contract():
    """Upload a contract to process"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save the file
        filename = file.filename
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        file.save(filepath)
        
        # Process the contract
        count = embedding_engine.process_contract_file(filepath)
        
        return jsonify({
            "message": f"Processed {count} clauses from {filename}",
            "filename": filename,
            "clause_count": count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/contracts', methods=['GET'])
def list_contracts():
    """Get list of all processed contracts"""
    try:
        contracts = embedding_engine.get_contracts_list()
        return jsonify({
            "contracts": contracts,
            "count": len(contracts)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure contracts directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
    
    # Initialize the embedding engine
    if embedding_engine.connect():
        embedding_engine.setup_database()
        embedding_engine.load_model()
    
    # Start the Flask app
    app.run(host=HOST, port=PORT, debug=True)