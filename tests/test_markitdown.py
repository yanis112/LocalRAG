from markitdown import MarkItDown

md = MarkItDown()
result = md.convert(r"data\sheets\Feuille 1.xlsx")
print(result.text_content)