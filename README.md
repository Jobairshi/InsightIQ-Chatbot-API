Here’s a **ready-to-paste** `README.md` file for your project:

````markdown
# IntelliFAQ API

> **A modern FastAPI backend for semantic FAQ search, intelligent chat, and AI-powered course & MCQ generation.**

---

## 🚀 Features
- **FAQ Search** – Fast semantic search using **TF-IDF + Cosine Similarity**  
- **AI Chat** – Context-aware chatbot leveraging **Azure OpenAI (GPT-4o)**  
- **Course Lesson Generator** – Create structured lessons with difficulty & duration controls  
- **MCQ Generator** – Auto-generate multiple-choice questions with customizable difficulty  
- **RESTful API** – Simple & powerful endpoints for integration  

---

## 🛠 Tech Stack
- **Backend:** Python 3.8+, FastAPI, Uvicorn  
- **AI:** Azure OpenAI (GPT-4o), LangChain  
- **Data:** Pandas, Scikit-learn  

---

## ⚡ Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/intellifaq-api.git
   cd intellifaq-api
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file and add:

   ```env
   AZURE_OPENAI_API_KEY=your_api_key_here
   ```

4. **Add your FAQ data**
   Place your FAQ dataset in `new_features.csv`

5. **Run the server**

   ```bash
   uvicorn faq_simple:app --host 0.0.0.0 --port 8001
   ```

---

## 📡 API Endpoints

* `POST /search-faq` – Search the FAQ knowledge base
* `POST /chat` – Chat with FAQ context integration
* `POST /generate-course-lesson` – Generate a course lesson
* `POST /generate-mcq` – Generate multiple-choice questions

---

## 📜 Example Response

```json
{
  "response": "The AI's answer to your query",
  "status": "success"
}
```

---

## 🔧 Troubleshooting

* **API Key Issues:** Ensure your Azure OpenAI key is valid & active.
* **Dependencies:** Run `pip install -r requirements.txt` again if errors occur.
* **Port Conflicts:** Change the port in `faq_simple.py` if `8001` is busy.

---
