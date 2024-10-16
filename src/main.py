from src.graph import app
from src.Tools import visualize_graph
from src.Databases.GraphDatabase import GraphDatabase
from src.Databases.config_defs import MainConfig
from src.LLM.config_defs import LLMMainConfig
from src.LLM.Pipeline import Pipeline

test_content = """
Ahmet, a student in the Mathematics Department at the university, often met Zeynep, who worked at the library. 
Zeynep was also an assistant to Professor Murat, who happened to be Ahmet’s instructor. Ahmet’s best friend, 
Mehmet, frequently gave him advice about Zeynep, and Zeynep also knew Mehmet from social gatherings. Meanwhile, 
it was well-known that Professor Murat had an old friendship with Ayşe, a member of the university board.
"""
# llm_config = LLMMainConfig.from_file("C:/Users/samit/Projects/langgraph-nl2kg/configs/LLM/openai.yaml")
#llm = Pipeline.new_instance_from_config(config=llm_config)
#result = llm.generate_kg_query(test_content)

result = app.invoke({"question": f"Can you generate a knowledge graph for the following text: {test_content}"})

