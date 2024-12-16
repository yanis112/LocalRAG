from src.main_utils.custom_langchain_componants import DoclingPDFLoader


def test_docling_pdf_loader():
    import time
    start = time.time()
    docling_pdf_loader = DoclingPDFLoader(file_path="aux_data/test.pdf")
    text = docling_pdf_loader.load()
    end_time = time.time()
    print(text) 
    print("#################")
    print("Time taken to load the pdf file:", end_time-start)

if __name__ == "__main__":
    test_docling_pdf_loader()