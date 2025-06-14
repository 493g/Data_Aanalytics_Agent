This repository holds the code for an AI Data Analyst Agent, an interactive web application built with Streamlit and powered by Together.ai's large language models. This agent lets you upload various types of documents, then use AI to analyze the data, answer your questions, and even generate insightful visualizations right in your browser.

Features
Versatile Document Support: Upload and process structured data like CSV and Excel (XLSX) files, or dive into unstructured text from TXT, PDF, DOCX, and even image files (PNG, JPG) with built-in OCR.
Intelligent Data Analysis: Ask the AI agent natural language questions about your uploaded data or document content, and get concise, relevant answers.
Automated Visualization: For your structured data, simply ask for visualizations (e.g., "show me the distribution of sales"). The agent will suggest and generate runnable Python code using Matplotlib and Seaborn to display charts directly within the app.
Interactive Chat Interface: Enjoy a seamless, conversational experience with the AI as you explore your data.
Technologies Used
Streamlit: For building the responsive and user-friendly web interface.
Together.ai API: Provides the powerful underlying large language models, specifically meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8.
Pandas: Essential for efficient data manipulation and analysis of structured datasets.
Matplotlib & Seaborn: The go-to libraries for creating high-quality data visualizations.
PyPDF2, python-docx, Pillow, pytesseract, pdf2image: Core libraries for document parsing and Optical Character Recognition (OCR) capabilities.
python-dotenv: For securely managing API keys and other environment variables.


Getting Started
Follow these steps to get your AI Data Analyst Agent up and running:

Clone the repository:

Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
(Remember to replace your-username and your-repo-name with your actual GitHub details.)

Set up your Python environment:
    It's highly recommended to use a virtual environment.

Bash

python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
Install Python dependencies:
Create a requirements.txt file in your project's root directory with the following content:

streamlit
pandas
numpy
matplotlib
seaborn
python-docx
openpyxl
PyPDF2
together
Pillow
pytesseract
pdf2image
python-dotenv
Then install them:

Bash

pip install -r requirements.txt
Configure your API Key:
Obtain your API key from Together.ai. Then, create a file named .env in the root directory of your project (the same folder as data_analyst_app.py) and add your key like this:

TOGETHER_API_KEY=your_actual_together_api_key
Replace your_actual_together_api_key with the key you obtained.

Install System Dependencies (for PDF/Image OCR):
This is a crucial step for the agent to process PDF and image files.

Tesseract OCR Engine:
Windows: Download and install from Tesseract-OCR GitHub. Make sure to add it to your system's PATH during installation, or manually.
macOS: brew install tesseract
Linux (Debian/Ubuntu): sudo apt-get install tesseract-ocr
Poppler (for pdf2image):
Windows: Download the latest release from Poppler for Windows. Extract it and add the bin directory inside the extracted folder to your system's PATH.
macOS: brew install poppler
Linux (Debian/Ubuntu): sudo apt-get install poppler-utils
Run the Streamlit app:
Once all dependencies are installed and your API key is set, launch the application from your terminal:

Bash

streamlit run data_analyst_app.py
This command will open the AI Data Analyst Agent in your default web browser.

Usage
Upload a Document: Use the "Upload Document" section in the sidebar to select your CSV, XLSX, TXT, DOCX, PDF, PNG, or JPG file. Click "Process File."
Ask Questions: Once the file is processed, type your questions about the data or document content into the chat input field.
Generate Visualizations: For structured data, simply type viz in the chat input. The agent will provide suggestions and display relevant charts.

