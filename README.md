# Data_Aanalytics_Agent
AI Data Analyst Agent
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


