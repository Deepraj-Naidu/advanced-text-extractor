import os
import streamlit as st
import fitz
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Get API key from Streamlit Secrets or .env
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = os.getenv("GEMINI_API_KEY")

# Streamlit UI
st.title("Advanced Invoice Text Extractor")
st.write("Upload an invoice PDF to extract important information.")

uploaded_file = st.file_uploader("Upload Invoice PDF", type=["pdf"])

# ------------------ Data Models ------------------

class BoundingBoxField(BaseModel):
    bounding_box: list[int] = Field(..., description='The bounding box where the information was found [y_min, x_min, y_max, x_max]')
    page: int = Field(..., description='Page number where the information was found. Start counting with 1.')

class TotalAmountField(BoundingBoxField):
    value: float = Field(..., description='The total amount of the invoice.')

class RecipientField(BoundingBoxField):
    name: str = Field(..., description='The name of the recipient.')

class TaxAmountField(BoundingBoxField):
    value: float = Field(..., description='The total amount of the tax.')

class SenderField(BoundingBoxField):
    name: str = Field(..., description='The name of the sender.')

class AccountNumberField(BoundingBoxField):
    account_no: str = Field(..., description='The number of the account.')

class InvoiceModel(BaseModel):
    total: TotalAmountField
    recipient: RecipientField
    tax: TaxAmountField
    sender: SenderField
    account_no: AccountNumberField

# ------------------ Run extraction after upload ------------------

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # Save uploaded file temporarily
    file_in = "uploaded_invoice.pdf"
    file_out = "invoice_annotated.pdf"

    with open(file_in, "wb") as f:
        f.write(uploaded_file.read())

    # Gemini client
    client = genai.Client(api_key=api_key)

    pdf = client.files.upload(file=file_in)

    prompt = """
    Extract the invoice recipient name and invoice total.
    Return ONLY JSON that matches the provided schema.
    If a field is missing, set it to null (and bounding_box to [0,0,0,0]).
    """

    with st.spinner("Analyzing invoice using Gemini AI..."):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pdf, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": InvoiceModel
            },
        )

    invoice = InvoiceModel.model_validate_json(response.text)

    st.subheader("Extracted Information")

    st.write({
        "Total Amount": invoice.total.value,
        "Recipient": invoice.recipient.name,
        "Tax": invoice.tax.value,
        "Sender": invoice.sender.name,
        "Account Number": invoice.account_no.account_no
    })

    # ------------------ Draw bounding boxes ------------------

    items_to_draw = [
        ("TOTAL", invoice.total.bounding_box, invoice.total.page),
        ("RECIPIENT", invoice.recipient.bounding_box, invoice.recipient.page),
        ("TAX", invoice.tax.bounding_box, invoice.tax.page),
        ("SENDER", invoice.sender.bounding_box, invoice.sender.page),
        ("ACCOUNT_NO", invoice.account_no.bounding_box, invoice.account_no.page)
    ]

    doc = fitz.open(file_in)

    for label, box, page_no in items_to_draw:
        if not box or box == [0, 0, 0, 0] or page_no is None:
            continue

        page = doc[page_no - 1]
        y0, x0, y1, x1 = box
        r = page.rect

        rect = fitz.Rect(
            (x0 / 1000) * r.width,
            (y0 / 1000) * r.height,
            (x1 / 1000) * r.width,
            (y1 / 1000) * r.height,
        )

        page.draw_rect(rect, color=(1, 0, 0), width=2)
        page.insert_text((rect.x0, rect.y0 - 2), label, fontsize=6, color=(1, 0, 0))

    doc.save(file_out)
    doc.close()

    st.success("Annotated PDF created!")

    with open(file_out, "rb") as f:
        st.download_button(
            label="Download Annotated Invoice",
            data=f,
            file_name="invoice_annotated.pdf",
            mime="application/pdf"
        )

else:
    st.info("Please upload an invoice PDF to continue.")

