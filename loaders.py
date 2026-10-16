from pathlib import Path
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

# to extract text from the pdf and return langchain documents
def load_pdf(file_path: str | Path) -> list[Document]:

    path=Path(file_path)

    if path.suffix.lower()!=".pdf":
        raise ValueError("File should be in pdf format")

    loader=DoclingLoader(
        file_path=str(path),
        export_type=ExportType.DOC_CHUNKS
    )
    docs=loader.load()

    if docs:
        print(f"Successfully processed {path.name} and chunked into {len(docs)} chunks.")
        print(f"Metadata keys in chunk 0: {list(docs[0].metadata.keys())}")

    return docs

