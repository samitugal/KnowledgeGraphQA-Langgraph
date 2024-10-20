import sys
import os

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
import streamlit.components.v1 as components
from src.graph import app
from src.Tools import visualize_graph
from src.Databases.GraphDatabase import GraphDatabase
from src.Databases.config_defs import MainConfig
from src.LLM.config_defs import LLMMainConfig
from src.LLM.Pipeline import Pipeline
from src.output_models import OperationType

def main():
    st.title("AI Application")

    # User input
    user_input = st.text_area("Please enter your text:", height=200)

    if st.button("Run"):
        if user_input:
            database_config = MainConfig.from_file(os.environ.get("GRAPH_DATABASE_CONNECTION_PATH"))
            llm_config = LLMMainConfig.from_file(os.environ.get("LLM_CONFIG_PATH"))

            pipeline = Pipeline.new_instance_from_config(config=llm_config)
            database = GraphDatabase.new_instance_from_config(config=database_config)

            result = app.invoke({"question": user_input})

            if result["operation_type"].operation_type == OperationType.GENERATE_KNOWLEDGE_GRAPH:
                st.subheader("Knowledge Graph:")
                try:
                    with open("graph.html", "r", encoding="utf-8") as f:
                        graph_html = f.read()
                    components.html(graph_html, height=600)
                except FileNotFoundError:
                    st.error("graph.html file not found.")
            else:
                st.subheader("Result:")
                st.write(result["result"].answer)

            database.disconnect()       
        else:
            st.warning("Please enter a text.")

if __name__ == "__main__":
    main()
