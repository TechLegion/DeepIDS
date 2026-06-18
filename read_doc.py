from docx import Document

doc = Document("Elizabeth's 1 -3..docx")
for para in doc.paragraphs:
    if para.text.strip():
        print(para.text)
