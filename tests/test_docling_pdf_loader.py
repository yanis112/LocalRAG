import time
import sys
from src.main_utils.custom_langchain_componants import DoclingPDFLoader

def test_docling_pdf_loader():
    # Execute the test
    start = time.time()
    docling_pdf_loader = DoclingPDFLoader(file_path="aux_data/test.pdf")
    text = docling_pdf_loader.load()
    end_time = time.time()
    execution_time = end_time - start
    
    # Print with immediate flush
    sys.stdout.write("\n\n=== TEST OUTPUT ===\n")
    sys.stdout.write(f"PDF Content (first 1000 chars):\n{text[:1000]}\n")
    sys.stdout.write("==================\n")
    sys.stdout.write(f"Time taken: {execution_time:.2f} seconds\n")
    sys.stdout.write("=== END OUTPUT ===\n\n")
    sys.stdout.flush()
    
    # Assertions
    assert text is not None, "PDF text should not be None"
    assert len(text) > 0, "PDF text should not be empty"
    
    
    
   
    