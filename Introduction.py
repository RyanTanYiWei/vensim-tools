import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="🔁",
)

st.write("# Suite of Vensim Internal Tools")

st.sidebar.success("Select Tools Above")

st.sidebar.markdown(
    """
    <!-- GitHub Repo Badge -->
    <a href="https://github.com/RyanTanYiWei/vensim-toolso" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-Repo-black?logo=github" alt="GitHub Repo">
    </a>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1800px; padding-left: 10rem; padding-right: 10rem;}
      /* Wrap long code blocks without changing colors */
      pre, code { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
      /* Add subtle separation around code blocks */
      pre { border: 1px solid rgba(0,0,0,0.08); border-radius: 6px; padding: 0.75rem; }

      /* Make metrics cards more card-like */
      [data-testid="stMetricValue"] {
          font-size: 1.5rem;
          font-weight: 700;
          color: #2e86de;
      }
      [data-testid="stMetricLabel"] {
          font-size: 0.9rem;
          font-weight: 500;
          opacity: 0.8;
      }

      /* Space and shadow for metric blocks */
      .css-1ht1j8u, .css-1r6slb0 {
          padding: 1rem !important;
          border-radius: 12px !important;
          box-shadow: 0 2px 6px rgba(0,0,0,0.08);
          background: #ffffff;
      }

      /* Section headers */
      h2, h3, .stSubheader {
          color: #1a5276;
          border-bottom: 1px solid rgba(0,0,0,0.1);
          padding-bottom: 0.3rem;
          margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
"""
This suite of tools is designed for large Vensim-based System Dynamics (SD) projects that are often contained in single files. These tools are meant to compensate for the inflexibilities of compartmentalization that arise from the strong visual elements of system dynamics models, as compared to traditional code-based models. By using these tools, you can enhance internal understanding 
of existing model versions as part of your workflow.

#### 👈 Key Tools
1. **⚖️ Version Comparison**  
   Compare two versions of a model at the level of their **mathematical formulation**. This ensures that any changes 
   affecting the model’s computations are detected, while purely visual or aesthetic changes—such as colors, 
   arrows, or repositioned variables in views—are ignored.

2. **🧩 Sub Model Analysis**  
   Typically, a large model uses multiple views to represent different modules/submodels as a way to conceptually 
   organize the complexity of the modelled system. This tool describes the **inter-module relationships**, as a means 
   to understand an existing model structure or to help in compartmentalising the model more intentionally.

3. **🕸 CLD to Network Conversion**  
   Convert causal loop diagrams (CLDs) from Vensim to typical **network formats** (i.e., node and edge lists). 
   This tool does not perform any analytical operations; it does only the "conversion", allowing visualization 
   and analysis to be conducted on other plaforms.

4. **📊 Vensim Plotter**  
   Transform Vensim simulation outputs into interactive, publication-ready charts. This flexible visualization tool 
   allows you to paste data directly from Excel or upload CSV files, create multi-dimensional groupings, apply 
   filters, and export charts for presentations and reports.
---

##### 🔮 Future Enhancements
- *(Sandeep)* **Improve colour schemes** in Version Comparison to make differences easier to spot.  
- *(YQL)* Allow **filtering of views for version comparison**
- *(Sandeep)* **Enable saving and uploading of parsed files** to reduce waiting times when re-running analyses.  
- *(Ryan)* **Provide useful network metrics** (e.g., centrality measures, feedback loop detection) in CLD to Network Conversion to enrich insights beyond structural export. 
"""

)
