import chromadb
import logging
import json
import os
import uuid
from pdfminer.high_level import extract_text
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import Generator, Optional, Union, Dict, List, Any
from fastchat.utils import build_logger

logger = build_logger('VectorDB', 'vector_db.log')
logger.setLevel(logging.DEBUG)


class VectorDB:
    def __init__(self, collection: str):
        # get current module path
        # https://stackoverflow.com/questions/247770/how-to-retrieve-a-modules-path

        self.chromadb_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                                        persist_directory='.chromadb'))
        # persist_directory="/path/to/persist/directory" # Optional, defaults to .chromadb/ in the current directory

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.chromadb_client.get_or_create_collection(collection,
                                                                        embedding_function=self.embedding_fn)
        self.slice_size = 400
        self.overlap = 40

    def add_doc(self, doc: List[str], meta: List[Dict], doc_id: List[str]) -> bool:
        """
            Add a document to the Vector database using slicing.
        """
        try:
            self.collection.add(documents=doc, metadatas=meta, ids=doc_id)
            return True
        except Exception as e:
            logger.exception(e)
            return False

    def query(self, query: str, **kwargs):
        """
            Query the Vector database.
        """
        try:
            results = self.collection.query(query_texts=query, **kwargs)
            return results
        except Exception as e:
            logger.exception(e)
            return None


def load_qa_into_vectordb(vdb, file_path: str) -> int:
    """
        Load data from a file into vector db. Each input line is a JSON record.
        {"quesiton": <>, "answer": <>, "district": <>}
        For each record, create three different vDB records:
        - answer field only
        - question field only
        - answer + question fields: answer + '\n\n' + question

        Meta data contains:
        - content_type = <question/answer/question+answer>
        - original_content = <original JSON content>
    """
    rid = 0
    with open(file_path, 'r') as f:
        for line in f:
            print(rid)
            record = json.loads(line)
            meta = {"content_type": "answer", "original_content": json.dumps(record)}
            vdb.add_doc([record['answer']], [meta], [str(rid)])
            rid += 1

            meta = {"content_type": "question", "original_content": json.dumps(record)}
            vdb.add_doc([record['question']], [meta], [str(rid)])
            rid += 1

            meta = {"content_type": "question+answer", "original_content": json.dumps(record)}
            vdb.add_doc([record['answer'] + '\n\n' + record['question']], [meta], [str(rid)])
            rid += 1

    logger.info(f'Loaded {rid} records into VectorDB.')
    return rid


def load_documents_into_vector_db(vdb, in_dir: str) -> bool:
    """
        Load all documents in a directory into VectorDB. Only the following types of files are processed:
        - .txt
        - .pdf
        Each file is treated as a document. If the input file is a pdf file, its content is extracted as text.
        Each file is split into trunks of text no larger than _slice_size. The overlap between two consecutive
        slices is _overlap.

        A UUID is generated as the document id. The meta data contains:
        - content_type: <txt/pdf>
        - source_file_name: source file name
    """
    _slice_size = 400
    _overlap = 40
    for file in os.listdir(in_dir):
        if file.endswith('.txt'):
            with open(os.path.join(in_dir, file), 'r') as f:
                text = f.read()
                meta = {"content_type": "txt", "source_file_name": file}
        elif file.endswith('.pdf'):
            pdf_file = os.path.join(in_dir, file)
            # extract text from pdf using pdfminer
            text = extract_text(pdf_file)
            meta = {"content_type": "pdf", "source_file": file}
        else:
            logger.warning(f'Unsupported file type: {file}')
            return False

        # trunk the document
        text_len = len(text)
        texts = []
        while text_len > _slice_size:
            texts.append(text[:_slice_size])
            text = text[_slice_size - _overlap:]
            text_len = len(text)
        if text_len > 0:
            texts.append(text)

        # add each trunk as a document
        for trunk in texts:
            # create a UUID as the document id
            doc_id = uuid.uuid4().hex
            vdb.add_doc([trunk], [meta], [doc_id])
    return True


if __name__ == '__main__':
    vdb = VectorDB('zhongke_qa')
    # load_qa_into_vectordb(vdb, '/Users/zhihengw/data/zhongke/json/zhongke_qa_v1.1.json')
    result = vdb.query('我想知道如何申请贷款', n_results=10)
    print(result)