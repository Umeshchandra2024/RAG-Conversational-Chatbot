from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.documents import Document

def load_documents(texts:str,pdf_paths:list[str]):
    docs=[]
    

    if texts:
        docs.extend([Document(page_content=texts,metadata={"source":"user_text"})])
    
    if pdf_paths:
        for pdf_path in pdf_paths:
            pdf_loader=PyPDFLoader(pdf_path)
            doc=pdf_loader.load()
            docs.extend(doc)
    
    return docs