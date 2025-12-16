import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ==========================================
# 1. SETUP & DATABASE
# ==========================================
st.set_page_config(page_title="Right Step Preschool Analytics", layout="wide")

def init_db():
    conn = sqlite3.connect('preschool_data.db', check_same_thread=False) 
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            student_name TEXT,
            q1_instructions TEXT,
            q2_turns TEXT,
            q3_emotions TEXT,
            q4_sharing TEXT,
            q5_space TEXT,
            q6_focus TEXT,
            q7_joining TEXT,
            q8_help TEXT,
            q9_independence TEXT,
            q10_persistence TEXT
        )
    ''')
    conn.commit()
    return conn

conn = init_db()

# ==========================================
# 2. SCORING FUNCTION
# ==========================================
def get_score(text):
    """Maps qualitative text responses to a quantitative score (1-5)."""
    mapping = {
        "Immediately": 5, "After 1 reminder": 4, "After several reminders": 3, "Rarely follows": 2, "Never follows": 1,
        "Always waits": 5, "Usually waits": 4, "Sometimes waits": 3, "Rarely waits": 2, "Never waits": 1,
        "Independent": 5, "Mostly independent": 4, "Sometimes needs help": 3, "Often loses control": 2, "Cannot regulate": 1,
        "Always shares": 5, "Usually shares": 4, "Sometimes shares": 3, "Rarely shares": 2, "Never shares": 1,
        "Always respects": 5, "Mostly respects": 4, "Sometimes respects": 3, "Rarely respects": 2, "Never respects": 1,
        "Highly focused": 5, "Mostly focused": 4, "Sometimes focused": 3, "Rarely focused": 2, "Cannot focus": 1,
        "Eagerly participates": 5, "Usually participates": 4, "Sometimes participates": 3, "Rarely participates": 2, "Does not participate": 1,
        "Asks independently": 5, "Asks after prompt": 4, "Sometimes asks": 3, "Rarely asks": 2, "Never asks": 1,
        "Completely independent": 5, "Mostly independent": 4, "Partially independent": 3, "Needs guidance": 2, "Not independent": 1,
        "Persistent": 5, "Mostly persistent": 4, "Sometimes gives up": 3, "Often gives up": 2, "Does not try": 1
    }
    return mapping.get(text, 0)

# ==========================================
# 3. CATEGORIES AND QUESTIONS
# ==========================================
categories = ["Instructions", "Turns", "Emotions", "Sharing", "Space",
              "Focus", "Joining", "Help", "Indep.", "Persistence"]

questions = {
    "q1_instructions": ("Follows Instructions: How consistently the child listens to and correctly follows teacher directions.", 
                        ["Immediately", "After 1 reminder", "After several reminders", "Rarely follows", "Never follows"]),
    "q2_turns": ("Takes Turns: How well the child waits for their turn during activities or discussions.", 
                 ["Always waits", "Usually waits", "Sometimes waits", "Rarely waits", "Never waits"]),
    "q3_emotions": ("Controls Emotions: How the child regulates emotions like frustration or anger.", 
                    ["Independent", "Mostly independent", "Sometimes needs help", "Often loses control", "Cannot regulate"]),
    "q4_sharing": ("Shares & Cooperates: Willingness to share toys and cooperate with peers.", 
                   ["Always shares", "Usually shares", "Sometimes shares", "Rarely shares", "Never shares"]),
    "q5_space": ("Maintains Personal Space: Respects others' personal space and classroom boundaries.", 
                 ["Always respects", "Mostly respects", "Sometimes respects", "Rarely respects", "Never respects"]),
    "q6_focus": ("Focuses on Tasks: Ability to maintain attention until completion.", 
                 ["Highly focused", "Mostly focused", "Sometimes focused", "Rarely focused", "Cannot focus"]),
    "q7_joining": ("Joins Activities: Willingness to participate in group activities.", 
                   ["Eagerly participates", "Usually participates", "Sometimes participates", "Rarely participates", "Does not participate"]),
    "q8_help": ("Asks for Help: Whether the child seeks help appropriately when needed.", 
                ["Asks independently", "Asks after prompt", "Sometimes asks", "Rarely asks", "Never asks"]),
    "q9_independence": ("Independent Work: Ability to perform tasks independently.", 
                        ["Completely independent", "Mostly independent", "Partially independent", "Needs guidance", "Not independent"]),
    "q10_persistence": ("Keeps Trying: How the child responds to challenging tasks.", 
                        ["Persistent", "Mostly persistent", "Sometimes gives up", "Often gives up", "Does not try"])
}

# ==========================================
# 4. CHART FUNCTIONS (Reusable)
# ==========================================

def create_radar_chart(scores, title, chart_color, label):
    """Generates a Matplotlib Radar Chart for the given scores."""
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += [angles[0]]
    data = scores + [scores[0]]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.plot(angles, data, linewidth=2, linestyle='solid', label=label, color=chart_color)
    ax.fill(angles, data, chart_color, alpha=0.25)
    
    plt.xticks(angles[:-1], categories, fontsize=10)
    ax.set_rlabel_position(30)
    plt.yticks([1, 2, 3, 4, 5], ["1","2","3","4","5"], color="grey", size=8)
    ax.set_ylim(0, 5)
    plt.title(title, size=12, y=1.1)
    
    return fig

def create_bar_chart(scores, title):
    """Generates a Matplotlib Horizontal Bar Chart for the given scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    percentages = [s / 5 * 100 for s in scores]
    colors = ['#4CAF50' if p >= 80 else '#FFC107' if p >= 60 else '#FF5252' for p in percentages]
    y_pos = np.arange(len(categories))
    
    ax.barh(y_pos, percentages, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Performance (%)")
    plt.title(title, size=12)
    
    for i, v in enumerate(percentages):
        ax.text(v + 1, i + 0.1, f"{v:.0f}%", fontweight='bold')
        
    return fig

# ==========================================
# 5. DATABASE MANAGEMENT FUNCTIONS (NEW)
# ==========================================

def delete_observation(obs_id):
    """Deletes a record from the database by ID."""
    c = conn.cursor()
    c.execute("DELETE FROM observations WHERE id = ?", (obs_id,))
    conn.commit()

def update_observation(obs_id, responses):
    """Updates an existing record in the database."""
    c = conn.cursor()
    
    # Construct the SET clause for the SQL query
    set_clauses = [f"{col} = ?" for col in responses.keys()]
    set_query = ", ".join(set_clauses)
    
    values = list(responses.values())
    values.append(obs_id)
    
    c.execute(f"UPDATE observations SET {set_query} WHERE id = ?", values)
    conn.commit()


# ==========================================
# 6. APP INTERFACE 
# ==========================================
tab_live, tab_student, tab_class = st.tabs(["🔥 Live Observation & Chart", "👤 Student Analysis & Management", "📊 Class Analysis"])

# --- TAB 1: LIVE OBSERVATION & CHART ---
with tab_live:
    st.title("Live Observation & Immediate Analysis")
    st.markdown("Record a new observation and see the student's performance profile instantly.")
    
    col_form, col_chart = st.columns([1, 1.5], gap="large")
    
    with col_form:
        st.markdown("### 📝 New Observation Entry")
        with st.form("live_entry_form"):
            student_name = st.text_input("Student Name", placeholder="e.g. Rohan Sharma", key="live_name")
            st.markdown("---")
            
            responses = {}
            for key, (label, options) in questions.items():
                st.write(f"**{label}**")
                responses[key] = st.radio("", options, horizontal=True, key=f"live_{key}", 
                                          label_visibility="collapsed")
            
            submit_button = st.form_submit_button("💾 Save Observation & Show Chart")

    with col_chart:
        st.markdown("### 📈 Live Performance Chart")
        
        if submit_button:
            if student_name.strip():
                # 1. Save Data to Database
                c = conn.cursor()
                cols = ", ".join(["student_name"] + list(responses.keys()))
                placeholders = ", ".join(["?"] * (len(responses) + 1))
                values = [student_name] + list(responses.values())
                c.execute(f"INSERT INTO observations ({cols}) VALUES ({placeholders})", values)
                conn.commit()
                st.success(f"✅ Observation for *{student_name}* saved successfully! Charts below reflect this submission.")
                
                # 2. Calculate Scores
                student_scores = [get_score(responses[col]) for col in questions.keys()]
                
                # 3. Display Charts instantly
                st.markdown("---")
                
                col_radar, col_bar = st.columns([1, 1.5])
                
                with col_radar:
                    radar_fig = create_radar_chart(student_scores, "Skill Profile (Radar Chart)", '#2196F3', student_name)
                    st.pyplot(radar_fig)

                with col_bar:
                    bar_fig = create_bar_chart(student_scores, "Performance Breakdown")
                    st.pyplot(bar_fig)

            else:
                st.error("⚠ Please enter a student name before saving.")
        else:
            st.info("Fill out the form and click 'Save Observation & Show Chart' to see the live results.")


# -------------------------------
# TAB 2: STUDENT ANALYSIS & MANAGEMENT (MODIFIED)
# -------------------------------
with tab_student:
    st.subheader("Individual Student Analysis & Management")
    df = pd.read_sql("SELECT * FROM observations", conn)
    
    if df.empty:
        st.info("Waiting for data... Please add observations first.")
    else:
        # Display list of all observations for the selected student
        students = df['student_name'].unique()
        selected_student = st.selectbox("Select Student", students, key="select_student_mgmt")
        
        # Get all observations for the student and sort by timestamp
        student_history = df[df['student_name'] == selected_student].sort_values(by='timestamp', ascending=False)
        
        st.markdown("---")
        
        # --- Observation Selector for Charting, Editing, and Deleting ---
        st.markdown("#### Select Observation for Charting/Management")
        
        # Create a display string for each observation (Timestamp and ID)
        history_options = {
            row['id']: f"{row['timestamp']} (ID: {row['id']})"
            for index, row in student_history.iterrows()
        }
        
        selected_id = st.selectbox("Choose an Observation:", list(history_options.keys()), 
                                   format_func=lambda x: history_options[x], 
                                   key="selected_obs_id")

        # Retrieve the selected observation row
        student_data_row = df[df['id'] == selected_id].iloc[0]
        timestamp_str = pd.to_datetime(student_data_row['timestamp']).strftime('%Y-%m-%d %H:%M')
        
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.markdown(f"**Showing Observation ID {selected_id}** (Date: {timestamp_str})")
            
            # --- DELETE BUTTON ---
            if st.button("❌ Delete This Observation", key="delete_button"):
                delete_observation(selected_id)
                st.success(f"Record ID {selected_id} for {selected_student} deleted. Please refresh the page to update the dashboard.")
                st.stop() # Stop execution to prompt a refresh
            
            # --- EDIT FORM ---
            st.markdown("#### ✏️ Edit Selected Observation")
            
            with st.form(f"edit_form_{selected_id}"):
                edit_responses = {}
                for key, (label, options) in questions.items():
                    # Get the current value from the dataframe row to set as default
                    current_value = student_data_row[key]
                    st.write(f"**{label}**")
                    # Set the index of the current value for the radio button default
                    default_index = options.index(current_value) if current_value in options else 0

                    edit_responses[key] = st.radio("", options, horizontal=True, 
                                                   index=default_index, 
                                                   key=f"edit_{key}_{selected_id}", 
                                                   label_visibility="collapsed")
                
                if st.form_submit_button("💾 Save Edited Observation"):
                    update_observation(selected_id, edit_responses)
                    st.success(f"Record ID {selected_id} updated successfully!")
                    st.rerun() # Rerun to refresh charts with new data

        # --- Chart Display for Selected Observation ---
        with col2:
            # Calculate scores
            score_cols = list(questions.keys())
            student_scores = [get_score(student_data_row[col]) for col in score_cols]
            
            st.markdown("#### Observation Charts")
            chart_col_1, chart_col_2 = st.columns([1, 1.5])
            
            with chart_col_1:
                radar_fig = create_radar_chart(student_scores, "Skill Profile (Radar)", '#2196F3', selected_student)
                st.pyplot(radar_fig)

            with chart_col_2:
                bar_fig = create_bar_chart(student_scores, "Performance Breakdown")
                st.pyplot(bar_fig)


# -------------------------------
# TAB 3: CLASS ANALYSIS (Overall Performance)
# -------------------------------
with tab_class:
    st.subheader("Overall Class Analysis")
    df = pd.read_sql("SELECT * FROM observations", conn)
    
    # This chart dynamically reflects ALL data, including new/edited/deleted observations.
    if df.empty:
        st.info("Waiting for data... Please add observations first.")
    else:
        avg_scores = [df[col].apply(get_score).mean() for col in questions.keys()]
        
        st.markdown("**Showing Class Average Across All Observations**")
        col_radar, col_bar = st.columns([1, 1.5])
        
        with col_radar:
            radar_fig = create_radar_chart(avg_scores, "Class Average Skill Profile", '#FF5252', "Class Average")
            st.pyplot(radar_fig)
            
        with col_bar:
            bar_fig = create_bar_chart(avg_scores, "Class Average Breakdown")
            st.pyplot(bar_fig)
            
        st.info("💡 Any changes (new, edited, or deleted observations) instantly update these Class Average charts.")