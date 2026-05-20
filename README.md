# 🧭 Vector Hunt Challenge

A interactive classroom web application built using **Streamlit** to teach students the fundamentals of vector spaces, text embeddings, semantic similarity, cosine similarity, vector databases, and Retrieval-Augmented Generation (RAG) through gamified gameplay.

Suitable for GenAI, Natural Language Processing (NLP), or general Data Science classroom workshops!

---

## 💡 What Students Learn

By participating in the **Vector Hunt Challenge**, students will master the following core concepts:
1. **Vectors and Embeddings:** Transforming natural language text into dense high-dimensional vectors (using the cached, local Sentence-Transformer model `all-MiniLM-L6-v2`).
2. **Cosine Similarity:** Understanding how models measure the angular alignment of meaning between vectors.
3. **Contextual Shift (Context Trap):** Discovering how the semantic representation of homonyms (e.g. *Apple*, *Bank*, *Java*) warps entirely based on adjacent context.
4. **Vector Search Engine Logic:** Encapsulating search queries and ranking matched documents based on nearest semantic distances rather than character overlaps.
5. **RAG & GenAI Pipelines:** Discovering how vector search matches custom documents dynamically to feed downstream Large Language Models (LLMs).

---

## 🚀 Quick Start - Local Installation

To run the application locally on your machine, follow these simple steps:

### 1. Clone or Copy the Project
Ensure your local project directory contains all necessary files:
```text
vector-hunt-challenge/
│
├── app.py
├── database.py
├── embedding_utils.py
├── challenges.py
├── auth_utils.py
├── genai_utils.py
├── students.json
├── requirements.txt
├── .env.example
└── README.md
```

### 2. Set Up a Virtual Environment
Navigate to the project root directory inside your shell:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all package requirements specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Create and Configure environmental variables
Copy `.env.example` to a new file named `.env` and set your administrator password:
```bash
# On Windows PowerShell:
copy .env.example .env

# On macOS / Linux:
cp .env.example .env
```
Inside `.env`, you can customize your admin password:
```text
ADMIN_PASSWORD=change_this_to_a_secure_password
```
*(If no .env is created, the system defaults to the secure fallback password: **`Strongback@2026!`**).*

### 5. Launch the Streamlit App
Run the local dev server:
```bash
streamlit run app.py
```
This will automatically launch the web interface in your default browser (usually at `http://localhost:8501`). The SQLite file `vector_hunt.db` will be initialized automatically in the root folder upon startup.

---

## ☁️ Deploying on Streamlit Community Cloud

The application is structured to be entirely Streamlit Cloud-friendly:

1. **Push to GitHub:** Commit all files (excluding your `venv/`, `.env`, and the generated `vector_hunt.db` database file) to a public or private GitHub repository.
2. **Deploy on Streamlit:** Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**. Select your repository, branch, and set the entrypoint to `app.py`.
3. **Configure secrets:** Under the app's advanced settings, add the `ADMIN_PASSWORD` variable to the Streamlit secrets panel:
   ```toml
   ADMIN_PASSWORD = "your_secure_classroom_password"
   ```
4. **Offline Similarity Matching:** The core vector game uses the cached `sentence-transformers` model locally, meaning **no paid third-party API key** is needed for the instructor or students to play all four levels!

---

## 🎮 Game Rules & Level Explanations

To log in, students must enter their **registered roll number** and **registered name** exactly as configured in [students.json](file:///d:/Desktop/AI-ML-Materials/Gen-AI/Projects/vector-hunt-challenge/students.json). Names are matched in a case-insensitive and whitespace-insensitive manner.

### Level 1: Word Similarity Challenge
* **Goal:** Understand that words representing similar concepts map close to each other in vector spaces.
* **Target Word:** e.g., `doctor`
* **Action:** Input exactly 10 unique, single words related to the target.
* **Scoring:** Words scoring a Cosine Similarity $\ge 0.60$ are classified as **Correct**. Getting at least **7 / 10** correct wins the level.
* **Formula:** $\text{Score} = \text{Average Cosine Similarity} \times 100$ (capped at 100%).

### Level 2: Sentence Similarity Challenge
* **Goal:** Discover semantic similarity beyond exact keywords (different words expressing the same concept).
* **Target Sentence:** e.g., `I want to buy a budget phone with a good camera.`
* **Action:** Input exactly 10 unique rephrased sentences that preserve this intent.
* **Scoring:** Sentences scoring a Cosine Similarity $\ge 0.65$ are classified as **Correct**. Getting at least **7 / 10** correct wins the level.

### Level 3: Context Trap Challenge
* **Goal:** Prove how the surrounding context shifts a word's position in vector space.
* **Target Sentence:** e.g., `Apple is a sweet fruit used in juice and salads.` vs. `Apple launched a new iPhone with better camera features.`
* **Action:** Input 10 words aligned strictly with the expected meaning context. (e.g., in the fruit context, food terms pass, whereas tech terms fail).
* **Scoring:** Cosine Similarity $\ge 0.65$ is **Correct**. At least **7 / 10** correct wins the level.

### Level 4: Mini Vector Database Search Challenge
* **Goal:** Learn how vector databases index and match documents against incoming search queries.
* **Gameplay:** Students are given a user query and must select the best semantic document match from a database table containing 10 indexed documents.
* **Scoring:**
  * Choosing the correct target document that ranks **Top-1** in actual cosine similarity yields **10 points**.
  * Choosing the correct target document that ranks in the **Top-3** yields **6 points**.
  * Incorrect choices yield **0 points**.
  * The final score is calculated as the sum of points across rounds converted to a percentage out of 100%.

---

## 🔑 Optional GenAI Feedback

To enable personalized feedback from the AI:
1. In the sidebar, students can select their preferred model provider: **OpenAI**, **Google Gemini**, or **Groq**.
2. They enter their temporary API key in the password-masked input and click **Save Key**.
3. After completing any level, a button called **"Generate AI Feedback"** or **"Explain Vector Search Result"** appears.
4. The backend calls the LLM securely using the session key, formats the prompt, and extracts structural, educational feedback on their vector layouts.
5. The API key is kept purely in Streamlit's temporary browser state (`st.session_state`) and is **never** printed, written to logs, or saved in SQLite.

---

## ⚙️ Teacher Dashboard & Suspicious Activity Detection

The Admin Panel is locked behind the Administrator Password. It features a complete telemetry control center:
1. **Class Scoreboard:** Ranks students based on the sum of their best scores across all 4 levels. Contains a CSV export.
2. **Audited Login Attempts:** Full login logs highlighting when students entered invalid roll numbers or wrong names.
3. **Suspicious Events Log:** Evaluates classroom activities in real-time using built-in SQLite heuristics to flag:
   - **Brute-forcing:** A single roll number targeted with more than 3 wrong names.
   - **Session-hopping:** A single browser session trying to log in as multiple roll numbers.
   - **Account-sharing:** A single roll number logging in from multiple distinctive browser fingerprints.
   - **Shared Devices:** A single browser fingerprint used by multiple roll numbers.
   - **Velocity Violation:** A student completing levels in unusually fast intervals (under 15 seconds), indicating copy-pasting.

### ⚠️ Important Telemetry Limitation Note:
> [!CAUTION]
> The suspicious activity heuristics are based on logs, browser user-agents, and session state parameters. They act as **"Suspicious activity evidence"** rather than **"Confirmed cheating proof"**. Streamlit sandboxing and standard browser sandboxing prevent identifying the physical individual behind a device with 100% mathematical certainty. Always interpret these charts as investigative guidelines for the classroom rather than absolute proof.

### Database Controls:
Teachers can completely reset/wipe the database by typing `RESET` into the admin confirmation box. This action drops all SQLite tables and clears all history to prepare for the next class session.
