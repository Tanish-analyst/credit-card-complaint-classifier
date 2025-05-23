# 🏦 Credit Card Complaint Classifier �  

A **BERT-based NLP model** that automatically categorizes credit card complaints into predefined classes with a confidence threshold.  

🔗 **Live Demo**: (https://credit-card-complaint-classifier.streamlit.app/)

---

## 📌 Overview  

This project fine-tunes a **BERT model** to classify credit card complaints into:  
- **Billing**  
- **Trouble using card**  
- **Features**  
- **Customer service**  
- **Fraud**  

Achieves **93% accuracy** on test data and **88%** on a separate evaluation dataset.  

---

## 🛠️ Tech Stack  

- **Model**: `bert-base-uncased` (fine-tuned via HuggingFace `transformers`)  
- **Backend**: FastAPI (REST API for predictions)  
- **Frontend**: Streamlit (user-friendly interface)  
- **Deployment**:**Render**  
- **Data**: Synthetic dataset (~2000 samples) generated via LLMs  

---
