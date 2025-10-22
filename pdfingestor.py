import re
from pypdf import PdfReader
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PDFIngestor:
    @staticmethod
    def read_pdf_file(pdf_filename: str) -> str:
        pdf_reader = PdfReader(pdf_filename)
        print(f"PDF {pdf_filename} loaded found {len(pdf_reader.pages)} pages")
        raw_text = ''
        # attach the title and author metadata if available so the LLM can tell what book this is.
        if pdf_reader.metadata.title is None or pdf_reader.metadata.author is None:
            print(f"PDF {pdf_filename} has missing or incomplete metadata.")
        else:
            raw_text = raw_text + 'title: ' + pdf_reader.metadata.title + ' by ' + pdf_reader.metadata.author + ' \n '
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            raw_text += page_text
        return raw_text

    @staticmethod
    def clean_extracted_text(text: str) -> str:
        # replace multiple whitespace with single space
        cleaned = re.sub(r'\s+', ' ', text)
        # weird control characters
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
        # leading trailing whitespace
        return cleaned.strip()

    @staticmethod
    def chunk_text(text: str) -> List[str]:
        chunk_size = 1000
        chunk_overlap = 200
        # Set up our text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks
