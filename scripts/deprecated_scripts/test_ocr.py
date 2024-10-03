from doctr.models import ocr_predictor
from doctr.io import DocumentFile

model = ocr_predictor(det_arch='db_resnet50', reco_arch='vitstr_base', pretrained=True)

single_img_doc = DocumentFile.from_images("test_factures/test_screen.png")

result = model(single_img_doc)

print(result)