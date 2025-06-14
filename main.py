# data_analyst_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
from docx import Document
import PyPDF2
from PIL import Image
import pytesseract
import together
from pdf2image import convert_from_path
import os
import tempfile
from dotenv import load_dotenv

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="AI Data Analyst")

# --- Configuration and API Key Setup ---
load_dotenv() # Load environment variables from .env file

TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY environment variable not found.")
    st.info("Please set the TOGETHER_API_KEY in your environment or in a `.env` file (e.g., TOGETHER_API_KEY=your_key_here).")
    st.stop() # Stop the app if the API key is not set

# Initialize Together.ai client (cached to avoid re-initializing on every rerun)
@st.cache_resource
def get_together_client():
    return together.Together(api_key=TOGETHER_API_KEY)

client = get_together_client()

# --- Helper Functions for Document Loading ---

def extract_text_from_doc(filepath):
    """Extracts text from a .doc (or .docx) file."""
    try:
        doc = Document(filepath)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        return f"Error extracting text from DOCX/DOC: {e}"

def extract_text_from_pdf(filepath):
    """
    Extracts text from a .pdf file.
    Tries PyPDF2 first, then falls back to OCR if no text is extracted.
    Requires `pdf2image` and `poppler-utils` for OCR fallback.
    """
    text = ""
    try:
        # Attempt text extraction with PyPDF2 first
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()

        # If PyPDF2 extracted no meaningful text, try OCR
        if not text.strip():
            st.info(f"PyPDF2 extracted no readable text from this PDF. Attempting OCR...")
            ocr_text = []
            # For Windows users, you might need to set poppler_path here:
            # from pdf2image.exceptions import PopplerNotInstalledError
            # try:
            #     images = convert_from_path(filepath, dpi=300, poppler_path=r'C:\path\to\poppler\bin')
            # except PopplerNotInstalledError:
            #     st.error("Poppler is not installed or not in PATH. OCR for PDF will not work.")
            #     return "Error: Poppler is required for PDF OCR."
            images = convert_from_path(filepath, dpi=300)
            for i, img in enumerate(images):
                try:
                    page_text = pytesseract.image_to_string(img)
                    ocr_text.append(page_text)
                except Exception as ocr_e:
                    st.warning(f"  Error during OCR on page {i+1}: {ocr_e}")
                    ocr_text.append("")
            text = "\n".join(ocr_text)

        return text

    except Exception as e:
        return f"Error during PDF processing (PyPDF2 or OCR): {e}"

def load_data(uploaded_file):
    """
    Loads data from various Streamlit UploadedFile types into appropriate formats.
    Returns (data, data_type_string) or (error_message, "error").
    """
    if uploaded_file is None:
        return None, "unsupported"

    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_content = io.BytesIO(uploaded_file.getvalue())

    # Use NamedTemporaryFile for files that require a disk path (like docx, pdf, images for OCR)
    temp_filepath = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filepath = temp_file.name

        if file_extension == 'csv':
            return pd.read_csv(file_content), "structured_data"
        elif file_extension == 'xlsx':
            try:
                xls = pd.ExcelFile(file_content)
                if len(xls.sheet_names) > 1:
                    st.warning(f"Excel file '{uploaded_file.name}' contains multiple sheets. Only the first sheet ('{xls.sheet_names[0]}') will be loaded.")
                return pd.read_excel(xls, sheet_name=xls.sheet_names[0]), "structured_data"
            except Exception as e:
                return f"Error reading XLSX file: {e}", "error"
        elif file_extension == 'txt':
            try:
                return file_content.read().decode('utf-8'), "unstructured_text"
            except Exception as e:
                return f"Error reading TXT file (check encoding?): {e}", "error"
        elif file_extension == 'doc':
            return extract_text_from_doc(temp_filepath), "unstructured_text"
        elif file_extension == 'pdf':
            extracted_pdf_text = extract_text_from_pdf(temp_filepath)
            if "Error during PDF processing" in extracted_pdf_text:
                return extracted_pdf_text, "error"
            elif not extracted_pdf_text.strip():
                return "PDF processed, but no readable text was extracted even with OCR (might be a completely blank or unreadable document).", "unstructured_text"
            else:
                return extracted_pdf_text, "unstructured_text"
        elif file_extension in ['png', 'jpg', 'jpeg']:
            if pytesseract: # Check if pytesseract module is available (meaning tesseract is installed/found)
                try:
                    img = Image.open(file_content)
                    # For Windows users, you might need to set tesseract_cmd here:
                    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    text = pytesseract.image_to_string(img)
                    if not text.strip():
                        return "Image processed with OCR, but no readable text was found.", "unstructured_text"
                    return text, "unstructured_text"
                except pytesseract.TesseractNotFoundError:
                    return "Tesseract OCR engine not found. Cannot extract text from image. Please ensure it's installed and in your PATH.", "error"
                except Exception as e:
                    return f"Error processing image with OCR: {e}", "error"
            else:
                return "Image file detected. OCR library (pytesseract) not available. Cannot extract text content.", "image_info"
        else:
            return None, "unsupported"
    finally:
        # Ensure temporary file is cleaned up
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# --- Data Analyst Agent Class Definition ---

class DataAnalystAgent:
    def __init__(self, together_client):
        self.client = together_client
        # Use st.session_state for persistence across reruns
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'data_type' not in st.session_state:
            st.session_state.data_type = None
        if 'data_description' not in st.session_state:
            st.session_state.data_description = ""
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def upload_document(self, uploaded_file):
        """
        Loads the document and prepares it for analysis, updating session state.

        Args:
            uploaded_file (streamlit.runtime.uploaded_file_manager.UploadedFile): The file object from st.file_uploader.

        Returns:
            str: A status message regarding the upload.
        """
        if uploaded_file is None:
            st.session_state.data = None
            st.session_state.data_type = None
            st.session_state.data_description = "No file uploaded."
            st.session_state.uploaded_file_name = None
            return "No file provided for upload."

        loaded_data, data_type = load_data(uploaded_file)

        if data_type in ["error", "unsupported"] or loaded_data is None:
            st.session_state.data = None
            st.session_state.data_type = "error"
            st.session_state.data_description = loaded_data if isinstance(loaded_data, str) else \
                                                f"File '{uploaded_file.name}' could not be processed. Data loader returned None or an error."
            st.session_state.uploaded_file_name = None
            return f"Upload failed: {st.session_state.data_description}"

        st.session_state.data = loaded_data
        st.session_state.data_type = data_type
        st.session_state.uploaded_file_name = uploaded_file.name

        if st.session_state.data_type == "structured_data":
            if not isinstance(st.session_state.data, pd.DataFrame):
                st.session_state.data = None
                st.session_state.data_type = "error"
                st.session_state.data_description = "Internal error: Expected DataFrame for structured data but got non-DataFrame. Check load_data function."
                return f"Upload failed: {st.session_state.data_description}"

            try:
                buffer = io.StringIO()
                st.session_state.data.info(buf=buffer)
                info_string = buffer.getvalue()

                st.session_state.data_description = (
                    f"Data Head:\n{st.session_state.data.head().to_string()}\n\n"
                    f"Columns: {', '.join(st.session_state.data.columns)}\n\n"
                    f"Data Info (Non-Null Counts & Dtypes):\n{info_string}"
                )
            except Exception as e:
                st.session_state.data_description = f"Structured data loaded, but failed to extract detailed info for LLM: {e}. Data might be empty or malformed."
                st.session_state.data_type = "error"
                st.session_state.data = None
                return f"Upload partially successful (data loaded), but description failed: {st.session_state.data_description}"

        elif st.session_state.data_type == "unstructured_text":
            st.session_state.data_description = st.session_state.data[:3000] + ("..." if len(st.session_state.data) > 3000 else "")
        elif st.session_state.data_type == "image_info":
            st.session_state.data_description = st.session_state.data

        return f"Document '{st.session_state.uploaded_file_name}' uploaded successfully as {st.session_state.data_type}."

    def _send_to_llama(self, prompt, system_message="You are a highly intelligent data analyst. You can analyze data, answer questions, and create visualizations."):
        """
        Sends a prompt to the Llama-4-Maverick model and returns its response.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
                timeout=120
            )
            return response.choices[0].message.content
        except together.TogetherError as e: # More general Together.ai API error catch
            return f"Together.ai API Error: {e}"
        except Exception as e: # General fallback for any other unexpected errors
            return f"An unexpected error occurred during LLM communication: {type(e).__name__} - {e}"

    def analyze_data(self, question=None):
        """
        Analyzes the loaded data based on the question or provides a general summary.
        """
        if st.session_state.data is None or st.session_state.data_type in ["error", "unsupported"]:
            return "Please upload a valid document first. " + st.session_state.data_description

        if st.session_state.data_type == "structured_data":
            prompt = f"You have been provided with the following structured data. Here is its schema and a glimpse:\n\n{st.session_state.data_description}\n\n"
            if question:
                prompt += f"Based on this data, please answer the following question: '{question}' Provide your answer directly and concisely, supporting with data points if possible."
            else:
                prompt += "Please provide a high-level summary, identify key insights (trends, outliers, relationships), and suggest potential areas for further analysis from this data. Be thorough."
        elif st.session_state.data_type == "unstructured_text":
            prompt = f"You have been provided with the following text content from a document:\n\n{st.session_state.data_description}\n\n"
            if question:
                prompt += f"Based on this text, please answer the following question: '{question}' Provide your answer directly and concisely."
            else:
                prompt += "Please summarize the key points, main topics, and any significant entities or figures mentioned in this document. Identify any potential structured data points or insights that could be extracted."
        else: # image_info
            if question:
                prompt = f"I have detected an image file. I cannot directly 'see' the image content. However, if your question '{question}' can be answered based on the file's name or known context, please provide a speculative answer. Otherwise, state that direct image analysis is not possible."
            else:
                return "This is an image file. I cannot perform detailed data analysis without OCR or a description of its content. Please ask a specific question if you believe there's information I can interpret from its metadata or known context."

        return self._send_to_llama(prompt)

    def suggest_visualizations(self, question=None):
        """
        Suggests appropriate visualizations for structured data and provides Python code.
        """
        if st.session_state.data_type != "structured_data" or st.session_state.data is None:
            return "Visualizations are primarily applicable to structured data (CSV, XLSX). Please upload such a file first."

        base_prompt = f"Given the following structured data. Here is its schema and a glimpse:\n\n{st.session_state.data_description}\n\n"
        if question:
            base_prompt += f"Considering the question: '{question}', what are the most appropriate types of visualizations (e.g., bar chart, line chart, scatter plot, histogram) and which columns should be used for the axes? Provide specific and complete matplotlib/seaborn Python code examples to generate these plots. Ensure the code does NOT include `plt.show()`. Generate one plot per code block. If multiple plots are suitable, provide each in its own separate code block, wrapped in ```python tags and no other text."
        else:
            base_prompt += "Suggest the most insightful visualizations for this data to reveal key trends, distributions, or relationships. Provide one or two complete matplotlib/seaborn Python code examples for the most insightful plots. Ensure the code does NOT include `plt.show()`. Generate one plot per code block. If multiple plots are suitable, provide each in its own separate code block, wrapped in ```python tags and no other text."

        llama_response = self._send_to_llama(
            base_prompt,
            system_message="You are a data visualization expert. Provide clear, runnable Python code for visualizations using matplotlib and seaborn based on the provided data context. Only output code, no conversational text."
        )

        # The response should ideally already be in code blocks due to the prompt
        # We will parse all of them
        code_blocks = re.findall(r"```python\n(.*?)```", llama_response, re.DOTALL)
        if code_blocks:
            return "\n".join(code_blocks) # Return all blocks found
        else:
            return f"LLM did not provide a valid Python code block for visualization. Raw response: {llama_response}"


    def generate_visualization(self, viz_code_suggestion):
        """
        Executes Python code for visualization and returns a base64 encoded image URI.
        """
        if st.session_state.data_type != "structured_data" or st.session_state.data is None:
            return "Cannot generate visualizations without structured data."

        try:
            buf = io.BytesIO()
            plt.ioff() # Turn off interactive plotting
            fig = plt.figure(figsize=(10, 6)) # Create a new figure for the plot

            exec_globals = {
                'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
                'data': st.session_state.data, # Provide the loaded DataFrame to the executed code
                '__builtins__': { # Restrict built-in functions for security
                    'print': print, 'len': len, 'range': range, 'str': str, 'int': int, 'float': float,
                    'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                    'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
                }
            }
            # Remove potentially dangerous built-in functions from exec_globals for security
            for restricted_builtin in ['__import__', 'exec', 'eval', 'compile', 'getattr', 'setattr', 'delattr', 'open']:
                 if restricted_builtin in exec_globals['__builtins__']:
                     del exec_globals['__builtins__'][restricted_builtin]

            exec(viz_code_suggestion, exec_globals, {})

            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig) # Close the figure to free memory and prevent it from showing up outside Streamlit
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            return f"Error generating visualization: {type(e).__name__} - {e}. Please check the provided code or ask for another suggestion."

# --- Streamlit Application Layout and Logic ---

def main():
    # Initialize the agent in session state
    if 'agent' not in st.session_state:
        st.session_state.agent = DataAnalystAgent(client)

    # Sidebar for file upload
    st.sidebar.header("Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "txt", "doc", "pdf", "jpg", "jpeg", "png"],
        key="file_uploader"
    )

    process_file_button = st.sidebar.button("Process File", key="process_file_button")

    # Logic to process file only if a new file is uploaded or button is clicked
    if uploaded_file and (process_file_button or uploaded_file.name != st.session_state.get('last_processed_file_name')):
        # Clear chat history on new file upload or explicit re-processing
        st.session_state.chat_history = []
        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment, especially for PDFs/images with OCR."):
            upload_status = st.session_state.agent.upload_document(uploaded_file)
            st.session_state.chat_history.append({"role": "system", "content": f"**Upload Status:** {upload_status}"})
            st.session_state.last_processed_file_name = uploaded_file.name # Keep track of the last processed file
            st.rerun() # Rerun to update the display based on new data

    # Display current document info in sidebar
    if st.session_state.data_type and st.session_state.data_type != "error" and st.session_state.uploaded_file_name:
        st.sidebar.markdown(f"---")
        st.sidebar.subheader("Current Document")
        st.sidebar.write(f"**File:** {st.session_state.uploaded_file_name}")
        st.sidebar.write(f"**Type:** `{st.session_state.data_type.replace('_', ' ').title()}`")
        with st.sidebar.expander("Show Data Info/Preview"):
            if st.session_state.data_type == "structured_data":
                st.text_area("Data Info (for LLM)", value=st.session_state.data_description, height=200, disabled=True, label_visibility="collapsed")
            else:
                st.text_area("Document Preview (for LLM)", value=st.session_state.data_description, height=200, disabled=True, label_visibility="collapsed")
    else:
        st.sidebar.info("Please upload a document to begin analysis.")

    # Chat Interface
    st.subheader("Chat with the Data Analyst")

    # Display chat messages from history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "agent":
            with st.chat_message("assistant"):
                # Check if the content is a base64 image URI for display
                if message["content"] and message["content"].startswith("data:image/png;base64,"):
                    st.image(message["content"], caption="Generated Visualization", use_column_width=True)
                else:
                    st.write(message["content"])
        elif message["role"] == "system":
            with st.chat_message("system"):
                st.info(message["content"])

    # User input chat box
    user_query = st.chat_input("Ask a question about the data, or type 'viz' for visualizations...", key="chat_input")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Handle queries based on data availability and type
        if st.session_state.data is None or st.session_state.data_type in ["error", "unsupported"]:
            response = "Please upload a valid document first before asking questions or requesting visualizations."
            st.session_state.chat_history.append({"role": "agent", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
        elif user_query.lower() == 'viz':
            if st.session_state.data_type == "structured_data":
                with st.spinner("Getting visualization suggestions..."):
                    viz_question = None # You could prompt for a specific question here if desired
                    viz_suggestion_text = st.session_state.agent.suggest_visualizations(viz_question)

                    # Display the raw code suggestion first (useful for debugging)
                    st.session_state.chat_history.append({"role": "agent", "content": f"### Visualization Suggestions:\n```python\n{viz_suggestion_text}\n```"})
                    with st.chat_message("assistant"):
                        st.markdown(f"### Visualization Suggestions:\n```python\n{viz_suggestion_text}\n```")

                    if "```python" in viz_suggestion_text:
                        st.info("Attempting to generate plot(s) from the suggested code block(s)...")

                        # Extract all code blocks (LLM might provide multiple or a single block with multiple plots)
                        code_blocks = re.findall(r"```python\n(.*?)```", viz_suggestion_text, re.DOTALL)

                        if code_blocks:
                            for i, code_block_raw in enumerate(code_blocks):
                                # 1. Clean the code: Remove plt.show() as it interferes with saving
                                cleaned_code_block = code_block_raw.replace("plt.show()", "").strip()

                                if not cleaned_code_block: # Skip if the block became empty after cleaning
                                    st.warning(f"Skipping empty or malformed code block {i+1}.")
                                    continue

                                # 2. Generate and display each visualization
                                plot_result_uri = st.session_state.agent.generate_visualization(cleaned_code_block)

                                if plot_result_uri.startswith("data:image"):
                                    st.session_state.chat_history.append({"role": "agent", "content": plot_result_uri})
                                    with st.chat_message("assistant"):
                                        st.image(plot_result_uri, caption=f"Generated Visualization {i+1}", use_column_width=True)
                                else:
                                    st.session_state.chat_history.append({"role": "agent", "content": f"Error generating plot {i+1}: {plot_result_uri}"})
                                    with st.chat_message("assistant"):
                                        st.error(f"Error generating plot {i+1}: {plot_result_uri}")

                            if not code_blocks: # This part is mostly for clarity if somehow regex fails
                                st.session_state.chat_history.append({"role": "agent", "content": "No executable Python code block found in the suggestions to plot."})
                                with st.chat_message("assistant"):
                                    st.warning("No executable Python code block found in the suggestions to plot.")
                        else:
                            st.session_state.chat_history.append({"role": "agent", "content": "No executable Python code block found in the suggestions to plot."})
                            with st.chat_message("assistant"):
                                st.warning("No executable Python code block found in the suggestions to plot.")
                    else:
                        st.session_state.chat_history.append({"role": "agent", "content": "No code block found to execute for visualization."})
                        with st.chat_message("assistant"):
                            st.warning("No code block found to execute for visualization.")
            else:
                response = "Visualizations are only applicable to structured data (CSV, XLSX)."
                st.session_state.chat_history.append({"role": "agent", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
        else: # Handle regular questions
            with st.spinner("Analyzing data..."):
                response = st.session_state.agent.analyze_data(user_query)
                st.session_state.chat_history.append({"role": "agent", "content": response})
                with st.chat_message("assistant"):
                    st.write(response)
        st.rerun() # Rerun to ensure the chat history is updated immediately and immediately show the latest message

if __name__ == '__main__':
    main()