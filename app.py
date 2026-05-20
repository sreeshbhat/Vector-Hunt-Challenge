# app.py
"""
Vector Hunt Challenge - Core Streamlit Application
A classroom gamified web app to teach embeddings, cosine similarity, and vector databases.
Designed to be Streamlit Cloud-friendly, secure, and highly educational.
"""

import time
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# Import our custom modules
import auth_utils
import database
import embedding_utils
import challenges
import genai_utils

# Load environment variables
load_dotenv()

# =====================================================================
# PAGE CONFIGURATION & STYLING
# =====================================================================
st.set_page_config(
    page_title="Vector Hunt Challenge",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Theme Custom CSS
st.markdown("""
<style>
    /* Premium fonts and headers */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 5px;
        margin-bottom: 20px;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #9CA3AF;
        margin-top: -15px;
        margin-bottom: 30px;
        font-size: 1.15rem;
    }
    
    .card-container {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .concept-title {
        color: #818CF8;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Outfit', sans-serif;
        color: #38BDF8;
    }
    
    /* Highlight vectors */
    .vector-label {
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        background-color: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        padding: 2px 6px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "session_id" not in st.session_state:
    st.session_state["session_id"] = auth_utils.generate_session_id()
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "roll_number" not in st.session_state:
    st.session_state["roll_number"] = ""
if "student_name" not in st.session_state:
    st.session_state["student_name"] = ""
if "class_section" not in st.session_state:
    st.session_state["class_section"] = ""
if "login_time" not in st.session_state:
    st.session_state["login_time"] = None
if "student_api_key" not in st.session_state:
    st.session_state["student_api_key"] = ""
if "ai_provider" not in st.session_state:
    st.session_state["ai_provider"] = "OpenAI"
if "db_initialized" not in st.session_state:
    database.init_db()
    st.session_state["db_initialized"] = True

# Gather telemetry markers once
if "client_ip" not in st.session_state:
    st.session_state["client_ip"] = auth_utils.get_client_ip_best_effort()
if "user_agent" not in st.session_state:
    st.session_state["user_agent"] = auth_utils.get_user_agent_best_effort()
if "device_fingerprint" not in st.session_state:
    st.session_state["device_fingerprint"] = auth_utils.generate_device_fingerprint(
        st.session_state["session_id"],
        st.session_state["user_agent"],
        st.session_state["client_ip"]
    )

# Maintain Level 4 persistent game board score states
if "l4_points" not in st.session_state:
    st.session_state["l4_points"] = {}  # maps query_index -> points earned

# Pre-load sentence transformers model cache immediately
try:
    embedding_utils.load_model()
except Exception:
    st.warning("⚠️ High-Dimensional Embedding model loading delay. App will load momentarily.")

# =====================================================================
# SIDEBAR CONTROLLER
# =====================================================================
with st.sidebar:
    st.markdown("## 🧭 Vector Hunt")
    st.markdown("---")
    
    # 1. Login Status Section
    if st.session_state["logged_in"]:
        st.success(f"Logged in: **{st.session_state['student_name']}**")
        st.markdown(f"**Roll Number:** `{st.session_state['roll_number']}`")
        st.markdown(f"**Section:** Class `{st.session_state['class_section']}`")
        if st.button("Logout", key="sidebar_logout_btn", use_container_width=True):
            # Safe Logout & Clean key
            st.session_state["logged_in"] = False
            st.session_state["roll_number"] = ""
            st.session_state["student_name"] = ""
            st.session_state["class_section"] = ""
            st.session_state["student_api_key"] = ""
            st.session_state["login_time"] = None
            st.session_state["l4_points"] = {}
            st.success("Successfully logged out.")
            time.sleep(0.5)
            st.rerun()
    else:
        st.info("🔴 Status: Guest. Please login in the first tab to save attempts & compete!")

    st.markdown("---")
    st.markdown("### 🤖 Optional GenAI Feedback")
    st.write("Unlock personalized tutor comments by providing a temporary API key. (Key exists only in session memory).")
    
    provider_sel = st.selectbox(
        "Select AI Provider",
        options=["OpenAI", "Google Gemini", "Groq"],
        index=0,
        key="sidebar_provider_sel"
    )
    st.session_state["ai_provider"] = provider_sel
    
    entered_key = st.text_input(
        "Enter your own API key",
        type="password",
        value=st.session_state["student_api_key"],
        placeholder=f"Your {provider_sel} API key...",
        help="Used ONLY to send semantic feedback queries. Never logged or stored in database."
    )
    
    col_key_1, col_key_2 = st.columns(2)
    with col_key_1:
        if st.button("Save Key", use_container_width=True):
            st.session_state["student_api_key"] = entered_key
            if entered_key.strip():
                st.success("Key saved for session!")
            else:
                st.warning("Empty key entered.")
    with col_key_2:
        if st.button("Clear Key", use_container_width=True):
            st.session_state["student_api_key"] = ""
            st.info("Key cleared.")
            time.sleep(0.5)
            st.rerun()

    st.markdown("🔒 *Safety Guarantee: Your key is kept inside active RAM (Streamlit session state) and is never printed, exposed in logs, or written to SQLite tables.*")

# =====================================================================
# MAIN DASHBOARD TABS
# =====================================================================
st.markdown("<h1 class='main-title'>🧭 Vector Hunt Challenge</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Master embeddings, semantic search, and vector spaces through gameplay</p>", unsafe_allow_html=True)

tab_names = [
    "👤 Student Login", 
    "🎯 Level 1: Words", 
    "📝 Level 2: Sentences", 
    "🎭 Level 3: Context Trap", 
    "🗄️ Level 4: Vector DB",
    "📊 My Results", 
    "🏆 Leaderboard", 
    "📖 Learn Concepts", 
    "⚙️ Admin Panel"
]

tabs = st.tabs(tab_names)

# ---------------------------------------------------------------------
# TAB 1: STUDENT LOGIN
# ---------------------------------------------------------------------
with tabs[0]:
    st.markdown("### 👤 Student Registry Verification")
    
    if st.session_state["logged_in"]:
        st.success(f"🎉 Welcome, **{st.session_state['student_name']}**! You have successfully authenticated.")
        
        col_wl_1, col_wl_2 = st.columns(2)
        with col_wl_1:
            st.markdown(f"""
            <div class='card-container'>
                <h4>Your Profile Card</h4>
                <p><b>Student Name:</b> {st.session_state['student_name']}</p>
                <p><b>Roll Number:</b> {st.session_state['roll_number']}</p>
                <p><b>Section Class:</b> {st.session_state['class_section']}</p>
                <p><b>Session Key:</b> <code>{st.session_state['session_id'][:8]}...</code></p>
                <p><b>Login Fingerprint:</b> <code>{st.session_state['device_fingerprint']}</code></p>
            </div>
            """, unsafe_allow_html=True)
        with col_wl_2:
            st.markdown("""
            #### 🚀 Gameplay Instructions:
            1. **Level 1 (Words):** Discover that semantically similar terms cluster close in space.
            2. **Level 2 (Sentences):** Learn that different word choices can yield near-identical sentence vector mappings.
            3. **Level 3 (Context Trap):** See how word meanings warp based on surrounding context.
            4. **Level 4 (Vector DB):** Act as a query encoder and search engine on a vector database.
            
            *Go to the tabs above to begin! Your score will be saved automatically.*
            """)
            
        if st.button("Logout Student Session", key="tab1_logout_btn"):
            st.session_state["logged_in"] = False
            st.session_state["roll_number"] = ""
            st.session_state["student_name"] = ""
            st.session_state["class_section"] = ""
            st.session_state["student_api_key"] = ""
            st.session_state["login_time"] = None
            st.session_state["l4_points"] = {}
            st.success("Successfully logged out.")
            time.sleep(0.5)
            st.rerun()
            
    else:
        st.write("Please sign in using your classroom roll number and full name as registered in `students.json`.")
        
        col_log_1, col_log_2 = st.columns(2)
        with col_log_1:
            with st.form("student_login_form"):
                inp_roll = st.text_input("Enter Roll Number", placeholder="e.g. 101").strip()
                inp_name = st.text_input("Enter Registered Name", placeholder="e.g. Rahul Kumar").strip()
                submit_login = st.form_submit_button("Verify & Enter Game", use_container_width=True)
                
                if submit_login:
                    if not inp_roll or not inp_name:
                        st.error("⚠️ Both roll number and name are required.")
                    else:
                        # Verify using auth_utils
                        res = auth_utils.verify_student_login(inp_roll, inp_name)
                        
                        if res["success"]:
                            student = res["student"]
                            st.session_state["logged_in"] = True
                            st.session_state["roll_number"] = student["roll_number"]
                            st.session_state["student_name"] = student["name"]
                            st.session_state["class_section"] = student["class_section"]
                            st.session_state["login_time"] = datetime.now()
                            
                            # Audit login
                            database.log_login_attempt(
                                inp_roll, inp_name, student["name"], student["class_section"],
                                True, "Login successful",
                                st.session_state["session_id"],
                                st.session_state["client_ip"],
                                st.session_state["user_agent"],
                                st.session_state["device_fingerprint"]
                            )
                            
                            # Heuristics scanning immediately
                            database.detect_suspicious_activity(
                                st.session_state["session_id"],
                                st.session_state["client_ip"],
                                st.session_state["user_agent"],
                                st.session_state["device_fingerprint"]
                            )
                            
                            st.success(f"🎉 Login Successful! Welcome {student['name']}.")
                            st.balloons()
                            time.sleep(1.0)
                            st.rerun()
                        else:
                            # Log failure
                            fail_student = res["student"]
                            sect = fail_student["class_section"] if fail_student else "unknown"
                            matched = fail_student["name"] if fail_student else "None"
                            
                            database.log_login_attempt(
                                inp_roll, inp_name, matched, sect,
                                False, res["reason"],
                                st.session_state["session_id"],
                                st.session_state["client_ip"],
                                st.session_state["user_agent"],
                                st.session_state["device_fingerprint"]
                            )
                            
                            # Run heuristics scanner on failures
                            database.detect_suspicious_activity(
                                st.session_state["session_id"],
                                st.session_state["client_ip"],
                                st.session_state["user_agent"],
                                st.session_state["device_fingerprint"]
                            )
                            
                            st.error(f"❌ Login Failed: {res['reason']}")
                            
        with col_log_2:
            st.markdown("""
            #### ⚙️ Student Registry Demo Credentials Example:
            - **Roll Number:** `23EG107D23` | **Name:** `Gouri Gundamoni` 
            
            *Name checking is space-insensitive and case-insensitive (e.g., '  rahul   kumar  ' works).*
            """)

# ---------------------------------------------------------------------
# HELPER FOR PLAYING LEVELS 1-3
# ---------------------------------------------------------------------
def play_semantic_challenge(level_number, level_name, challenge_list):
    st.markdown(f"### 🎯 Level {level_number}: {level_name}")
    
    if not st.session_state["logged_in"]:
        st.warning("⚠️ Access Denied. Please verify your credentials in the **Student Login** tab first.")
        return

    # Choose Target word/sentence
    targets_options = [c["target"] for c in challenge_list]
    selected_target = st.selectbox(
        "Select Challenge Target to hunt:",
        options=targets_options,
        key=f"lvl_{level_number}_target_select"
    )
    
    # Retrieve challenge dict
    chal = next(c for c in challenge_list if c["target"] == selected_target)
    threshold = chal["threshold"]
    min_correct = chal["minimum_correct"]
    
    # Render instructions
    st.markdown(f"""
    <div class='card-container'>
        <p><b>Target Semantic Vector:</b> <span class='vector-label'>{selected_target}</span></p>
        <p><b>Threshold Cosine Similarity:</b> <code>{threshold}</code> ({int(threshold*100)}%)</p>
        <p><b>Requirement:</b> Get at least <b>{min_correct} / 10 correct matches</b> to win this level!</p>
        <p><i>{chal.get('instructions', '')}</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Multiline text input for 10 entries
    input_text_area = st.text_area(
        "Enter 10 lines of text (exactly one per line, no duplicates):",
        height=220,
        placeholder="Input 1\nInput 2\n...\nInput 10",
        key=f"lvl_{level_number}_input_area"
    )
    
    submit_btn = st.button("Submit Vectors", key=f"lvl_{level_number}_submit_btn", type="primary")
    
    if submit_btn:
        # Start timer
        start_time = time.time()
        
        # Process entries
        raw_lines = input_text_area.split("\n")
        cleaned_inputs = [line.strip() for line in raw_lines if line.strip()]
        
        # Validation checks
        if len(cleaned_inputs) != 10:
            st.error(f"🛑 Validation Failed: You entered **{len(cleaned_inputs)}** items. Please enter **exactly 10** non-empty lines.")
            return
            
        if len(set(cleaned_inputs)) != len(cleaned_inputs):
            st.error("🛑 Validation Failed: Duplicate items detected! All 10 entries must be unique to map distinctive vector coords.")
            return
            
        with st.spinner("Analyzing semantic vectors and computing high-dimensional distances..."):
            # Compute embeddings and prepare results
            res_df = embedding_utils.prepare_similarity_results(selected_target, cleaned_inputs, threshold)
            
            if res_df.empty:
                st.error("An error occurred during vector comparisons.")
                return
                
            # Compute gameplay stats
            avg_sim = float(res_df["similarity_score"].mean())
            correct_cnt = int(res_df["is_correct"].sum())
            won = correct_cnt >= min_correct
            time_taken = time.time() - start_time
            
            # Score logic: Cap average similarity * 100 at 100
            score = round(max(0.0, avg_sim * 100), 2)
            score = min(100.0, score)
            
            # Save results into database
            attempt_id = database.save_attempt(
                st.session_state["roll_number"],
                st.session_state["student_name"],
                st.session_state["class_section"],
                st.session_state["session_id"],
                level_number,
                level_name,
                selected_target,
                score,
                avg_sim,
                correct_cnt,
                10,
                won,
                time_taken
            )
            
            # Save granular attempt items
            items_to_save = []
            for _, row in res_df.iterrows():
                items_to_save.append({
                    "input_text": row["input_text"],
                    "similarity_score": row["similarity_score"],
                    "is_correct": row["is_correct"]
                })
            database.save_attempt_items(attempt_id, items_to_save)
            
            # Run suspicious logs scanner
            database.detect_suspicious_activity(
                st.session_state["session_id"],
                st.session_state["client_ip"],
                st.session_state["user_agent"],
                st.session_state["device_fingerprint"]
            )
            
            # Cache results in st.session_state for optional AI feedback integration
            st.session_state[f"lvl_{level_number}_last_results"] = {
                "target": selected_target,
                "df": res_df,
                "score": score,
                "avg_sim": avg_sim,
                "correct_cnt": correct_cnt,
                "won": won
            }
            
            if won:
                st.balloons()
                
    # Render results if cached
    result_cache_key = f"lvl_{level_number}_last_results"
    if result_cache_key in st.session_state:
        res = st.session_state[result_cache_key]
        rdf = res["df"]
        
        # Display feedback banner
        if res["won"]:
            st.success(f"🏆 **Level Cleared!** You achieved **{res['correct_cnt']}/10** correct matches with an average similarity of **{round(res['avg_sim']*100, 2)}%**! (Score: **{res['score']}**)")
        else:
            st.error(f"💔 **Try Again!** You achieved **{res['correct_cnt']}/10** correct matches. You need at least **{min_correct}** to win this level.")
            
        # Display Metrics Dashboard
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Score", f"{res['score']} pts")
        with col_m2:
            st.metric("Correct Matches", f"{res['correct_cnt']} / 10")
        with col_m3:
            st.metric("Avg Cosine Similarity", f"{round(res['avg_sim'], 4)}")
        with col_m4:
            st.metric("Level Status", "🏆 WON" if res["won"] else "❌ TRY AGAIN")
            
        # Visual Layout
        col_lay_1, col_lay_2 = st.columns([1, 1])
        
        with col_lay_1:
            st.markdown("#### 📊 Vector Match Scorecard")
            # Style classification results
            display_df = rdf.copy()
            display_df["result"] = display_df["is_correct"].apply(lambda x: "🟢 Correct" if x == 1 else "🔴 Weak Match")
            display_df = display_df.rename(columns={
                "input_text": "Student Input",
                "similarity_percentage": "Cosine Similarity (%)",
                "result": "Classification"
            })
            st.dataframe(
                display_df[["Student Input", "Cosine Similarity (%)", "Classification"]],
                use_container_width=True,
                hide_index=True
            )
            st.info("💡 Cosine similarity measures the directional alignment of vectors. Identical concepts score near 100%, while unrelated ones drift lower.")

            # Bar Chart
            st.markdown("#### 📊 Semantic Alignments (Cosine similarity %)")
            fig_bar = px.bar(
                rdf,
                x="input_text",
                y="similarity_percentage",
                color="is_correct",
                color_continuous_scale=[[0, "#F87171"], [1, "#34D399"]],
                labels={"similarity_percentage": "Similarity %", "input_text": "Your Inputs", "is_correct": "Correct"},
                title="Input Vectors Distance from Target"
            )
            fig_bar.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.caption("Each bar reflects how close your input vector aligns to the target. Green indicates passing scores.")

        with col_lay_2:
            st.markdown("#### 🗺️ 2D Semantic Embedding Space Map")
            
            # Combine target and inputs for PCA mapping
            pca_texts = [res["target"]] + rdf["input_text"].tolist()
            pca_embeddings = embedding_utils.get_embeddings(pca_texts)
            pca_coords = embedding_utils.reduce_embeddings_pca(pca_embeddings)
            
            # Build PCA Plot Dataframe
            plot_data = []
            # Target row
            plot_data.append({
                "Text": f"🎯 TARGET: {res['target']}",
                "X": pca_coords[0, 0],
                "Y": pca_coords[0, 1],
                "Type": "Target",
                "Size": 15
            })
            # Student input rows
            for idx, row in rdf.iterrows():
                plot_data.append({
                    "Text": row["input_text"],
                    "X": pca_coords[idx + 1, 0],
                    "Y": pca_coords[idx + 1, 1],
                    "Type": "Correct" if row["is_correct"] == 1 else "Weak Match",
                    "Size": 10
                })
            
            pdf = pd.DataFrame(plot_data)
            
            fig_scatter = px.scatter(
                pdf,
                x="X",
                y="Y",
                color="Type",
                color_discrete_map={"Target": "#FBBF24", "Correct": "#34D399", "Weak Match": "#F87171"},
                size="Size",
                hover_name="Text",
                title="PCA Dimensionally Reduced Embedding Coordinates"
            )
            # Custom markers
            fig_scatter.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("💡 *Real embeddings operate in 384 dimensions. This chart reduces them to 2D using PCA (Principal Component Analysis) to visualize semantic clusters. Cluttered coordinates represent matching contexts!*")
            
            # Optional GenAI feedback section
            st.markdown("#### 🤖 Antigravity AI Tutor Feedback")
            
            # Retrieve session api key details
            api_key = genai_utils.get_student_api_key()
            provider = st.session_state.get("ai_provider", "OpenAI")
            
            if api_key and api_key.strip():
                if st.button("🤖 Generate AI Feedback", key=f"lvl_{level_number}_ai_feedback_btn"):
                    with st.spinner("Asking tutor to examine vector alignment..."):
                        ai_res = genai_utils.generate_ai_feedback(
                            provider, api_key, f"Level {level_number}: {level_name}",
                            res["target"], rdf
                        )
                        
                        st.session_state[f"lvl_{level_number}_ai_cached"] = ai_res
                
                ai_cache_key = f"lvl_{level_number}_ai_cached"
                if ai_cache_key in st.session_state:
                    ai_data = st.session_state[ai_cache_key]
                    
                    st.markdown(f"""
                    <div style='background-color: rgba(99, 102, 241, 0.1); border-left: 4px solid #818CF8; padding: 15px; border-radius: 8px;'>
                        <h5>Tutor Summary</h5>
                        <p>{ai_data.get('summary', '')}</p>
                        <p><b>🌟 Best Semantic Matches:</b> {', '.join(ai_data.get('best_matches', []))}</p>
                        <p><b>⚠️ Weak Semantic Matches:</b> {', '.join(ai_data.get('weak_matches', []))}</p>
                        <p><b>🧭 Concepts Taught:</b> {ai_data.get('concept_explanation', '')}</p>
                        <p><b>💡 Improvement Tips:</b></p>
                        <ul>
                            {''.join(f"<li>{tip}</li>" for tip in ai_data.get('improvement_tips', []))}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🔑 **Unlock Tutor Insights:** Enter your own LLM API key in the sidebar and click **Save Key** to enable diagnostic AI feedback.")

# ---------------------------------------------------------------------
# TAB 2: LEVEL 1 (Word Similarity)
# ---------------------------------------------------------------------
with tabs[1]:
    play_semantic_challenge(1, "Word Similarity Challenge", challenges.LEVEL_1_CHALLENGES)

# ---------------------------------------------------------------------
# TAB 3: LEVEL 2 (Sentence Similarity)
# ---------------------------------------------------------------------
with tabs[2]:
    play_semantic_challenge(2, "Sentence Similarity Challenge", challenges.LEVEL_2_CHALLENGES)

# ---------------------------------------------------------------------
# TAB 4: LEVEL 3 (Context Trap)
# ---------------------------------------------------------------------
with tabs[3]:
    play_semantic_challenge(3, "Context Trap Challenge", challenges.LEVEL_3_CHALLENGES)

# ---------------------------------------------------------------------
# TAB 5: LEVEL 4 (Vector Database Search)
# ---------------------------------------------------------------------
with tabs[4]:
    st.markdown("### 🗄️ Level 4: Mini Vector Database Search Challenge")
    
    if not st.session_state["logged_in"]:
        st.warning("⚠️ Access Denied. Please verify your credentials in the **Student Login** tab first.")
    else:
        st.markdown("""
        <div class='card-container'>
            <p><b>Goal:</b> Act as a vector search engine! You are given a semantic user query and a database of 10 items.
            Select the item that the vector search model will rank closest using <b>Cosine Similarity</b>.</p>
            <p><b>Scoring Rules:</b></p>
            <ul>
                <li>Selecting the correct expected item which ranks <b>Top-1</b> yields <b>10 points</b>.</li>
                <li>Selecting the correct expected item which ranks in the <b>Top-3</b> yields <b>6 points</b>.</li>
                <li>Wrong selections yield <b>0 points</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display database items
        st.markdown("#### 📂 Mock Vector Database (`VECTOR_DB_ITEMS`)")
        db_df = pd.DataFrame(challenges.VECTOR_DB_ITEMS)
        st.dataframe(db_df.rename(columns={"text": "Stored Text", "label": "Dataset Label"}), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Dropdown to choose which query to run
        selected_q_idx = st.selectbox(
            "Select Query Round to solve:",
            options=range(len(challenges.LEVEL_4_CHALLENGES)),
            format_func=lambda idx: f"Round {idx + 1}: '{challenges.LEVEL_4_CHALLENGES[idx]['query']}'"
        )
        
        active_q = challenges.LEVEL_4_CHALLENGES[selected_q_idx]
        query_text = active_q["query"]
        expected_label = active_q["expected_label"]
        
        st.info(f"🔍 **Active Student Search Query:** \"{query_text}\"")
        
        # Input selection
        label_options = [item["label"] for item in challenges.VECTOR_DB_ITEMS]
        student_selection = st.selectbox(
            "Select the database item that has the highest semantic similarity:",
            options=label_options,
            key=f"l4_selection_q_{selected_q_idx}"
        )
        
        execute_btn = st.button("Execute Vector Search", key=f"l4_run_btn_{selected_q_idx}", type="primary")
        
        if execute_btn:
            # Start timer
            l4_start_time = time.time()
            
            with st.spinner("Encoding query, querying vector database, and sorting embeddings..."):
                # Encode query
                query_emb = embedding_utils.get_embedding(query_text)
                
                # Encode database items
                db_texts = [item["text"] for item in challenges.VECTOR_DB_ITEMS]
                db_embs = embedding_utils.get_embeddings(db_texts)
                
                # Compute similarities
                sim_scores = embedding_utils.calculate_cosine_similarity(query_emb, db_embs)
                
                # Create ranked dataframe
                ranks_data = []
                for idx, item in enumerate(challenges.VECTOR_DB_ITEMS):
                    ranks_data.append({
                        "label": item["label"],
                        "text": item["text"],
                        "similarity_score": float(sim_scores[idx]),
                        "similarity_percentage": round(float(sim_scores[idx]) * 100, 2)
                    })
                rdf = pd.DataFrame(ranks_data).sort_values(by="similarity_percentage", ascending=False).reset_index(drop=True)
                # Assign rank numbers (1-indexed)
                rdf["rank"] = rdf.index + 1
                
                # Find rank of expected item and student selected item
                expected_row = rdf[rdf["label"] == expected_label].iloc[0]
                expected_rank = int(expected_row["rank"])
                
                student_row = rdf[rdf["label"] == student_selection].iloc[0]
                student_rank = int(student_row["rank"])
                
                # Score evaluation
                pts = 0
                is_correct = 0
                if student_selection == expected_label:
                    is_correct = 1
                    if expected_rank == 1:
                        pts = 10
                    elif expected_rank <= 3:
                        pts = 6
                
                # Update persistent state score
                st.session_state["l4_points"][selected_q_idx] = pts
                
                # Save attempt into attempts & granular items
                l4_time_taken = time.time() - l4_start_time
                
                # Overall Level 4 current percentage score
                total_earned = sum(st.session_state["l4_points"].values())
                max_possible = len(challenges.LEVEL_4_CHALLENGES) * 10
                current_pct_score = round((total_earned / max_possible) * 100, 2)
                
                attempt_id = database.save_attempt(
                    st.session_state["roll_number"],
                    st.session_state["student_name"],
                    st.session_state["class_section"],
                    st.session_state["session_id"],
                    4,
                    "Mini Vector Database Search Challenge",
                    query_text,
                    current_pct_score,
                    float(rdf["similarity_score"].mean()),
                    1 if is_correct else 0,
                    1,
                    1 if is_correct else 0,
                    l4_time_taken
                )
                
                # Save granular item details (ranking database matches)
                items_to_save = []
                for _, row in rdf.iterrows():
                    items_to_save.append({
                        "input_text": row["text"],
                        "expected_match": row["label"],
                        "similarity_score": row["similarity_score"],
                        "is_correct": 1 if row["label"] == expected_label else 0,
                        "rank_position": int(row["rank"])
                    })
                database.save_attempt_items(attempt_id, items_to_save)
                
                # Heuristics updates
                database.detect_suspicious_activity(
                    st.session_state["session_id"],
                    st.session_state["client_ip"],
                    st.session_state["user_agent"],
                    st.session_state["device_fingerprint"]
                )
                
                # Cache round output
                st.session_state[f"l4_last_round_{selected_q_idx}_res"] = {
                    "rdf": rdf,
                    "expected_label": expected_label,
                    "student_selection": student_selection,
                    "student_rank": student_rank,
                    "expected_rank": expected_rank,
                    "pts": pts,
                    "explanation": active_q["explanation"]
                }
                
                if pts == 10:
                    st.balloons()
                    
        # Render round output if available
        round_cache_key = f"l4_last_round_{selected_q_idx}_res"
        if round_cache_key in st.session_state:
            res = st.session_state[round_cache_key]
            rrdf = res["rdf"]
            
            # Show points badge
            if res["pts"] == 10:
                st.success(f"🎉 **Perfect Match!** You earned **10 points**! Your choice is the Top-1 database match.")
            elif res["pts"] == 6:
                st.info(f"🟠 **Near Match!** You earned **6 points**. Your choice is in the Top-3 but not the ultimate top match.")
            else:
                st.error(f"❌ **Mismatch!** You earned **0 points**. Expected Target was **'{res['expected_label']}'** (Ranked #{res['expected_rank']}), but you selected **'{res['student_selection']}'** (Ranked #{res['student_rank']}).")
                
            # Scorecard status
            st.markdown(f"#### 🎯 Active Round Scorecard: **+{res['pts']} pts**")
            
            # Render Ranked Table
            display_rrdf = rrdf.copy()
            # Highlight matching categories
            def highlight_match(row):
                if row["label"] == res["expected_label"]:
                    return "🎯 Target Match"
                elif row["label"] == res["student_selection"]:
                    return "👤 Your Choice"
                return "📂 DB Item"
                
            display_rrdf["Status"] = display_rrdf.apply(highlight_match, axis=1)
            
            # Rename columns
            display_rrdf = display_rrdf.rename(columns={
                "rank": "Search Rank",
                "text": "Database Item",
                "similarity_percentage": "Similarity Score (%)",
                "label": "DB Item Key"
            })
            
            st.dataframe(
                display_rrdf[["Search Rank", "Database Item", "Similarity Score (%)", "Status"]],
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown(f"💡 **Embedding Explanation:** {res['explanation']}")
            
            col_l4_lay1, col_l4_lay2 = st.columns(2)
            
            with col_l4_lay1:
                # Bar Chart
                fig_l4_bar = px.bar(
                    rrdf,
                    x="label",
                    y="similarity_percentage",
                    color="similarity_percentage",
                    color_continuous_scale="Purples",
                    labels={"similarity_percentage": "Similarity %", "label": "DB Item Key"},
                    title="Vector Similarity Scores Across Entire Database"
                )
                st.plotly_chart(fig_l4_bar, use_container_width=True)
                st.caption("Cosine similarities across all items. The highest peak represents the best search retrieval matching vector.")

            with col_l4_lay2:
                # Scatter Plot
                st.markdown("#### 🗺️ Vector DB Search Embedding Space Map")
                # Combine query + database items for PCA
                pca_l4_texts = [query_text] + db_texts
                pca_l4_embs = embedding_utils.get_embeddings(pca_l4_texts)
                pca_l4_coords = embedding_utils.reduce_embeddings_pca(pca_l4_embs)
                
                # Build PCA Plot Dataframe
                plot_data = []
                plot_data.append({
                    "Text": f"👤 Query: {query_text}",
                    "X": pca_l4_coords[0, 0],
                    "Y": pca_l4_coords[0, 1],
                    "Type": "Query Search",
                    "Size": 15
                })
                
                for idx, item in enumerate(challenges.VECTOR_DB_ITEMS):
                    lbl = item["label"]
                    t_type = "Database Item"
                    if lbl == res["expected_label"]:
                        t_type = "Target Match"
                    elif lbl == res["student_selection"]:
                        t_type = "Your Choice"
                        
                    plot_data.append({
                        "Text": f"{lbl}: {item['text']}",
                        "X": pca_l4_coords[idx + 1, 0],
                        "Y": pca_l4_coords[idx + 1, 1],
                        "Type": t_type,
                        "Size": 10
                    })
                    
                pl4_df = pd.DataFrame(plot_data)
                
                fig_l4_scatter = px.scatter(
                    pl4_df,
                    x="X",
                    y="Y",
                    color="Type",
                    color_discrete_map={"Query Search": "#EC4899", "Target Match": "#FBBF24", "Your Choice": "#3B82F6", "Database Item": "#9CA3AF"},
                    size="Size",
                    hover_name="Text",
                    title="PCA Mapping: Search Query vs DB Items"
                )
                fig_l4_scatter.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
                st.plotly_chart(fig_l4_scatter, use_container_width=True)
                st.caption("How your query is positioned compared to all database documents in the semantic vector space.")

                # Optional LLM explanations
                st.markdown("#### 🤖 AI Search Query Explanation")
                api_key = genai_utils.get_student_api_key()
                provider = st.session_state.get("ai_provider", "OpenAI")
                
                if api_key and api_key.strip():
                    if st.button("🤖 Explain Vector Search Result", key=f"l4_ai_explain_{selected_q_idx}"):
                        with st.spinner("Generating conceptual explanations..."):
                            # Craft a brief mock dataframe to feed into feedback generator
                            mock_results = pd.DataFrame([
                                {"input_text": f"Your Selection: {res['student_selection']}", "similarity_percentage": rrdf[rrdf["label"] == res["student_selection"]].iloc[0]["similarity_percentage"], "is_correct": 1 if res["pts"] > 0 else 0},
                                {"input_text": f"Expected Selection: {res['expected_label']}", "similarity_percentage": rrdf[rrdf["label"] == res["expected_label"]].iloc[0]["similarity_percentage"], "is_correct": 1}
                            ])
                            ai_explain = genai_utils.generate_ai_feedback(
                                provider, api_key, "Level 4: Mini Vector Database Search",
                                query_text, mock_results
                            )
                            st.session_state[f"l4_ai_cached_{selected_q_idx}"] = ai_explain
                            
                    ai_cache_key = f"l4_ai_cached_{selected_q_idx}"
                    if ai_cache_key in st.session_state:
                        ex_data = st.session_state[ai_cache_key]
                        st.markdown(f"""
                        <div style='background-color: rgba(99, 102, 241, 0.1); border-left: 4px solid #818CF8; padding: 15px; border-radius: 8px;'>
                            <h5>Tutor Explains Matches</h5>
                            <p>{ex_data.get('summary', '')}</p>
                            <p><b>🧭 High Dimensional Insights:</b> {ex_data.get('concept_explanation', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🔑 **Unlock Search Insights:** Enter your API key in the sidebar to generate custom tutor search query expansions.")
                    
        # Total Overall Score Card
        st.markdown("---")
        st.markdown("#### 🏆 Overall Level 4 Scoreboard")
        total_earned = sum(st.session_state["l4_points"].values())
        max_possible = len(challenges.LEVEL_4_CHALLENGES) * 10
        st.metric("Level 4 Aggregate Points:", f"{total_earned} / {max_possible} points", f"{round((total_earned/max_possible)*100, 2)}% Completed")

# ---------------------------------------------------------------------
# TAB 6: MY RESULTS
# ---------------------------------------------------------------------
with tabs[5]:
    st.markdown("### 📊 Your Level Performance Ledger")
    
    if not st.session_state["logged_in"]:
        st.warning("⚠️ Access Denied. Please verify your credentials in the **Student Login** tab first.")
    else:
        st.markdown(f"**Viewing results for:** `{st.session_state['student_name']}` (Roll: `{st.session_state['roll_number']}`)")
        
        # Load attempts
        attempts_df = database.get_student_results(st.session_state["roll_number"])
        
        if attempts_df.empty:
            st.info("You have not submitted any level challenges yet. Let's make some attempts!")
        else:
            # Metrics Dashboard
            best_scores = {}
            for l in range(1, 5):
                level_df = attempts_df[attempts_df["level_number"] == l]
                best_scores[l] = float(level_df["score"].max()) if not level_df.empty else 0.0
                
            total_wins = int(attempts_df["won"].sum())
            total_attempts = len(attempts_df)
            
            st.markdown("#### 🎯 Personal Best Dashboard")
            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
            with col_b1:
                st.metric("Level 1 Best", f"{best_scores[1]} %")
            with col_b2:
                st.metric("Level 2 Best", f"{best_scores[2]} %")
            with col_b3:
                st.metric("Level 3 Best", f"{best_scores[3]} %")
            with col_b4:
                st.metric("Level 4 Best", f"{best_scores[4]} %")
            with col_b5:
                st.metric("Attempts / Wins", f"{total_attempts} / {total_wins}")
                
            st.markdown("---")
            st.markdown("#### 📋 Detailed Gameplay Logs")
            
            # Format display
            disp_attempts = attempts_df.copy()
            disp_attempts["won"] = disp_attempts["won"].apply(lambda x: "🏆 YES" if x == 1 else "❌ NO")
            disp_attempts["score"] = disp_attempts["score"].apply(lambda x: f"{x} %")
            disp_attempts["average_similarity"] = disp_attempts["average_similarity"].apply(lambda x: round(x, 4))
            disp_attempts["time_taken_seconds"] = disp_attempts["time_taken_seconds"].apply(lambda x: f"{round(x, 2)} s")
            
            st.dataframe(
                disp_attempts.rename(columns={
                    "level_number": "Level",
                    "level_name": "Level Topic",
                    "target_text": "Target Semantic Word/Sentence",
                    "score": "Score",
                    "average_similarity": "Avg Cosine Similarity",
                    "correct_count": "Matches",
                    "total_items": "Total Entries",
                    "won": "Completed",
                    "time_taken_seconds": "Duration",
                    "created_at": "Timestamp"
                }).drop(columns=["id"]),
                use_container_width=True,
                hide_index=True
            )

# ---------------------------------------------------------------------
# TAB 7: LEADERBOARD
# ---------------------------------------------------------------------
with tabs[6]:
    st.markdown("### 🏆 Public Classroom Leaderboard")
    
    col_lead_1, col_lead_2 = st.columns([4, 1])
    with col_lead_2:
        refresh_btn = st.button("🔄 Refresh Scoreboard", use_container_width=True)
        
        # Download csv button
        csv_data = database.export_leaderboard_data()
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="vector_hunt_leaderboard.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_lead_1:
        st.write("Score rankings are derived from the sum of the best scores from all 4 game levels per student.")
        
    # Get scoreboard
    scoreboard = database.get_leaderboard()
    
    if scoreboard.empty:
        st.info("No scores saved yet. Be the first to secure a place on the board!")
    else:
        # Add Rank number manually
        scoreboard.insert(0, "Rank", scoreboard.index + 1)
        
        # Format metrics
        scoreboard["l1_best"] = scoreboard["l1_best"].apply(lambda x: f"{round(x, 2)}%")
        scoreboard["l2_best"] = scoreboard["l2_best"].apply(lambda x: f"{round(x, 2)}%")
        scoreboard["l3_best"] = scoreboard["l3_best"].apply(lambda x: f"{round(x, 2)}%")
        scoreboard["l4_best"] = scoreboard["l4_best"].apply(lambda x: f"{round(x, 2)}%")
        scoreboard["total_score"] = scoreboard["total_score"].apply(lambda x: f"{round(x, 2)} pts")
        scoreboard["average_score"] = scoreboard["average_score"].apply(lambda x: f"{round(x, 2)}%")
        
        st.dataframe(
            scoreboard.rename(columns={
                "roll_number": "Roll",
                "student_name": "Student Name",
                "class_section": "Class",
                "l1_best": "Level 1 Word Best",
                "l2_best": "Level 2 Sentence Best",
                "l3_best": "Level 3 Context Best",
                "l4_best": "Level 4 DB Search Best",
                "total_score": "Aggregate Points",
                "average_score": "Average Score",
                "levels_attempted": "Levels Attempted",
                "wins": "Total Level Clears",
                "last_attempt_time": "Last Submission"
            }),
            use_container_width=True,
            hide_index=True
        )

# ---------------------------------------------------------------------
# TAB 8: LEARN CONCEPTS
# ---------------------------------------------------------------------
with tabs[7]:
    st.markdown("### 📖 High-Dimensional Semantic Spaces handbook")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown(f"""
        #### 1. What is a Vector?
        In data science, a **Vector** is an ordered list of numerical values representing a position in multi-dimensional space.
        Instead of simple values, vectors represent complex semantic parameters.
        For example, the word <i>'doctor'</i> might translate into a vector like:
        <span class='vector-label'>[0.24, -0.67, 0.81, ..., 0.05]</span>.
        
        #### 2. What is an Embedding?
        An **Embedding** is a mathematical vector representing the *semantic meaning* of an input (text, audio, image, etc.).
        Embedding models (like the cached <code>all-MiniLM-L6-v2</code> used locally in this game) translate human sentences into high-dimensional numerical models such that concepts with similar contexts map closer in direction.
        
        #### 3. What is Cosine Similarity?
        **Cosine Similarity** is a mathematical formula that calculates the cosine of the angle between two vectors in an inner product space.
        It evaluates the directional alignment rather than length:
        $$\\text{{Cosine Similarity}} = \\cos(\\theta) = \\frac{{\\mathbf{{A}} \\cdot \\mathbf{{B}}}}{{\\|\\mathbf{{A}}\\| \\|\\mathbf{{B}}\\|}}$$
        - **1.0 (100%):** Vectors point in the exact same direction (semantically identical).
        - **0.0 (0%):** Vectors are orthogonal (completely unrelated meaning).
        - **-1.0:** Vectors point in opposite directions.
        """)
        
        st.markdown("""
        #### 4. Keyword Search vs. Vector Search
        - **Keyword Search:** Looks for exact characters and letter combinations (e.g. searching 'cheap phone for photos' misses 'budget smartphone with excellent camera' because zero characters overlap).
        - **Vector Search (Semantic Search):** Encodes both the query and documents into high-dimensional embeddings and measures cosine distances. It matches **intent and concept**, successfully retrieving similar contexts.
        """)
        
    with col_c2:
        st.markdown("""
        #### 5. How this connects to RAG (Retrieval-Augmented Generation)?
        Retrieval-Augmented Generation (RAG) is a framework that allows Large Language Models (LLMs) to retrieve custom contextual data from custom documents dynamically.
        
        **Semantic RAG Flowchart:**
        ```text
        [Document File] 
              ↓
        [Text Chunker] (Split text into small pieces)
              ↓
        [Embedding Model] (Transform chunks into 384-dim Vectors)
              ↓
        [Vector Database] (Index and store embeddings)
              ↓
        [User Question] → [Encoded Search Query Vector]
                                  ↓
                        [Similarity Distance Match]
                                  ↓
                        [Retrieve Best Document Chunks]
                                  ↓
                        [LLM Context Prompt] → [Accurate AI Answer]
        ```
        
        #### 6. What is a Vector Database?
        A **Vector Database** (e.g. Pinecone, Chroma, Milvus) is a highly specialized database designed to store, index, and query vector embeddings.
        It uses approximate nearest neighbor algorithms (ANN) to locate matching vectors across millions of rows in milliseconds.
        
        #### 7. Optional API Key Integration
        We use the Sentence-Transformer library to perform all game-level cosine similarity checks completely offline.
        However, if you want an AI tutor to explain *why* some vectors were weak or how they compare to the target target conceptually, you can enter your temporary API key in the sidebar. This key is stored in your local browser sandbox memory and cleared when you log out.
        """)

# ---------------------------------------------------------------------
# TAB 9: ADMIN PANEL
# ---------------------------------------------------------------------
with tabs[8]:
    st.markdown("### ⚙️ Teacher Control Center & Telemetry Panel")
    
    # Password lock
    if "admin_authenticated" not in st.session_state:
        st.session_state["admin_authenticated"] = False
        
    if not st.session_state["admin_authenticated"]:
        st.write("Please authenticate with the classroom Administrator Password.")
        admin_pwd = st.text_input("Enter Administrator Password", type="password", key="admin_pwd_field")
        
        if st.button("Unlock Admin Tab", key="admin_pwd_submit"):
            if auth_utils.check_admin_password(admin_pwd):
                st.session_state["admin_authenticated"] = True
                st.success("Access Granted.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Invalid Administrator Password.")
    else:
        # Logout Admin button
        col_ad_hdr1, col_ad_hdr2 = st.columns([4, 1])
        with col_ad_hdr2:
            if st.button("🔐 Lock Panel", key="lock_admin_btn", use_container_width=True):
                st.session_state["admin_authenticated"] = False
                st.info("Locked.")
                time.sleep(0.5)
                st.rerun()
                
        # Warning banner if fallback password is in use
        if auth_utils.is_fallback_admin_password_active():
            st.warning("⚠️ **Security Warning:** The application is running on the fallback admin password. Set the `ADMIN_PASSWORD` variable in your Streamlit secrets or environment for production deployments.")
            
        admin_menu = st.selectbox(
            "Select Administrator Tool:",
            options=[
                "🚨 Academic Integrity & Suspicious Events Dashboard",
                "👤 Audited Failed Login Attempts",
                "📈 Complete Class Scoreboard & Leaderboard",
                "🔍 Search Student Attempts Ledger",
                "🗄️ Database reset & Settings"
            ]
        )
        
        st.markdown("---")
        
        # Admin tool 1: Suspicious events
        if admin_menu.startswith("🚨"):
            st.markdown("#### 🚨 Flagged Suspicious Activity Audit Log")
            st.write("These indicators highlight potential academic compromises using heuristic patterns. They act as *suspicious activity evidence*, not *confirmed cheating proof*.")
            
            # Export logs CSV
            se_csv = database.export_suspicious_events_data()
            st.download_button(
                label="📥 Export Suspicious Events CSV",
                data=se_csv,
                file_name="suspicious_events_log.csv",
                mime="text/csv"
            )
            
            events = database.get_suspicious_events()
            if events.empty:
                st.success("No anomalies detected by heuristic analyzers. Clean game session!")
            else:
                # Format severity highlights
                def format_severity(sev):
                    if sev == "High":
                        return "🔴 High"
                    elif sev == "Medium":
                        return "🟡 Medium"
                    return "🔵 Low"
                events["severity"] = events["severity"].apply(format_severity)
                
                st.dataframe(
                    events.rename(columns={
                        "roll_number": "Student Roll",
                        "event_type": "Flagged Marker",
                        "severity": "Severity Level",
                        "description": "Event Diagnostics",
                        "session_id": "Browser Session Key",
                        "ip_address": "Client IP",
                        "user_agent": "Browser User-Agent",
                        "device_fingerprint": "Session Signature",
                        "created_at": "Logged Time"
                    }).drop(columns=["id"]),
                    use_container_width=True,
                    hide_index=True
                )
                
        # Admin tool 2: Failed Login Audits
        elif admin_menu.startswith("👤"):
            st.markdown("#### 👤 Audited Failed Login Attempts")
            st.write("Tracks login mismatch records where credentials entered do not align to `students.json`.")
            
            # Export login audits CSV
            la_csv = database.export_login_logs_data()
            st.download_button(
                label="📥 Export Authentication Audits CSV",
                data=la_csv,
                file_name="failed_login_logs.csv",
                mime="text/csv"
            )
            
            raw_log = database.get_login_logs()
            if raw_log.empty:
                st.info("No login attempts audited.")
            else:
                failed_log = raw_log[raw_log["login_success"] == 0]
                if failed_log.empty:
                    st.success("Zero failed logins audited!")
                else:
                    st.dataframe(
                        failed_log.rename(columns={
                            "roll_number_entered": "Roll Inputted",
                            "name_entered": "Name Inputted",
                            "matched_registered_name": "Matched Reg Name",
                            "class_section": "Reg Section",
                            "failure_reason": "Rejection Diagnostics",
                            "session_id": "Session Key",
                            "ip_address": "Client IP",
                            "user_agent": "User-Agent",
                            "device_fingerprint": "Fingerprint",
                            "created_at": "Attempt Time"
                        }).drop(columns=["id", "login_success"]),
                        use_container_width=True,
                        hide_index=True
                    )
                    
        # Admin tool 3: Complete Scoreboard
        elif admin_menu.startswith("📈"):
            st.markdown("#### 🏆 Live Classroom Scores Dashboard")
            
            # Display Scoreboard
            scoreboard = database.get_leaderboard()
            if scoreboard.empty:
                st.info("No scores registered yet.")
            else:
                scoreboard.insert(0, "Rank", scoreboard.index + 1)
                
                # Format
                scoreboard["l1_best"] = scoreboard["l1_best"].apply(lambda x: f"{round(x, 2)}%")
                scoreboard["l2_best"] = scoreboard["l2_best"].apply(lambda x: f"{round(x, 2)}%")
                scoreboard["l3_best"] = scoreboard["l3_best"].apply(lambda x: f"{round(x, 2)}%")
                scoreboard["l4_best"] = scoreboard["l4_best"].apply(lambda x: f"{round(x, 2)}%")
                scoreboard["total_score"] = scoreboard["total_score"].apply(lambda x: f"{round(x, 2)} pts")
                scoreboard["average_score"] = scoreboard["average_score"].apply(lambda x: f"{round(x, 2)}%")
                
                st.dataframe(
                    scoreboard.rename(columns={
                        "roll_number": "Student Roll",
                        "student_name": "Registered Name",
                        "class_section": "Class Section",
                        "l1_best": "L1 (Word) Best",
                        "l2_best": "L2 (Sentence) Best",
                        "l3_best": "L3 (Context) Best",
                        "l4_best": "L4 (DB Search) Best",
                        "total_score": "Total Score",
                        "average_score": "Average Score",
                        "levels_attempted": "Levels Finished",
                        "wins": "Total Wins",
                        "last_attempt_time": "Last Submission"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
        # Admin tool 4: Search Student attempts
        elif admin_menu.startswith("🔍"):
            st.markdown("#### 🔍 Query Student Attempts Registry")
            search_roll = st.text_input("Filter results by Student Roll Number", placeholder="e.g. 101").strip()
            
            attempts = database.get_attempts_by_roll_number(search_roll)
            if attempts.empty:
                st.warning("No records matched your search query.")
            else:
                disp_att = attempts.copy()
                disp_att["won"] = disp_att["won"].apply(lambda x: "🏆 YES" if x == 1 else "❌ NO")
                disp_att["score"] = disp_att["score"].apply(lambda x: f"{x} %")
                disp_att["average_similarity"] = disp_att["average_similarity"].apply(lambda x: round(x, 4))
                disp_att["time_taken_seconds"] = disp_att["time_taken_seconds"].apply(lambda x: f"{round(x, 2)} s")
                
                st.dataframe(
                    disp_att.rename(columns={
                        "roll_number": "Roll",
                        "student_name": "Student Name",
                        "class_section": "Section",
                        "level_number": "Level",
                        "level_name": "Topic",
                        "target_text": "Target Context",
                        "score": "Score",
                        "average_similarity": "Avg Cosine Similarity",
                        "correct_count": "Matches",
                        "total_items": "Items Count",
                        "won": "Completed",
                        "time_taken_seconds": "Duration",
                        "created_at": "Timestamp"
                    }).drop(columns=["id", "session_id"]),
                    use_container_width=True,
                    hide_index=True
                )
                
        # Admin tool 5: Reset DB & settings
        elif admin_menu.startswith("🗄️"):
            st.markdown("#### 📊 Reset Score Board Only")
            st.info("💡 **Preserve Logs:** This resets all student scores, level completions, and gameplay attempts. It preserves all login audits and academic integrity/suspicious activity logs intact.")
            
            confirm_sb = st.text_input("To reset the scoreboard only, type 'RESET SCOREBOARD' in capital letters:", key="confirm_sb_input", placeholder="RESET SCOREBOARD")
            if st.button("🗑️ Reset Scoreboard Only", key="reset_sb_btn", type="secondary"):
                if confirm_sb == "RESET SCOREBOARD":
                    with st.spinner("Wiping scoreboard attempts..."):
                        database.reset_scoreboard()
                        st.session_state["l4_points"] = {}
                        st.success("Scoreboard reset successfully!")
                        time.sleep(1.0)
                        st.rerun()
                else:
                    st.error("❌ Action Blocked: Confirmation input mismatch.")
            
            st.markdown("---")
            
            st.markdown("#### 🗄️ Wipe Database & Re-Initialize Registry")
            st.warning("⚠️ **CRITICAL WARNING:** Resetting the database drops all attempts, login audit trails, leaderboard history, and suspicious event telemetry permanently. This action is irreversible.")
            
            confirm_word = st.text_input("To verify this wipe action, type the confirmation word strictly in CAPITAL LETTERS: 'RESET'", placeholder="RESET")
            
            if st.button("🔥 Confirm Database Wipe & Rebuild", type="primary"):
                if confirm_word == "RESET":
                    with st.spinner("Deleting SQLite tables and reconstructing database schema..."):
                        database.reset_database()
                        st.success("Database wiped and re-initialized successfully!")
                        st.session_state["l4_points"] = {}
                        time.sleep(1.0)
                        st.rerun()
                else:
                    st.error("❌ Action Blocked: Confirmation word mismatch.")
                    
            st.markdown("---")
            st.markdown("#### 📂 Student JSON Registry Control")
            st.write("Click below to reload/re-index the students registry from `students.json`.")
            
            if st.button("🔄 Reload registry index"):
                try:
                    auth_utils.load_students()
                    st.success("Successfully loaded students.json into memory index.")
                except Exception as e:
                    st.error(f"Failed to reload registry: {str(e)}")
