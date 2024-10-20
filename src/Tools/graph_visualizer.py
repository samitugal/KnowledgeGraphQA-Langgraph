import networkx as nx
from src.Databases.GraphDatabase import GraphDatabase
from pyvis.network import Network
import streamlit as st

def create_graph(graph_database: GraphDatabase):
    data = graph_database.fetch_all()
    G = nx.DiGraph()

    # Iterate over all records to extract nodes and relationships
    for record in data:
        n = record[0]  # First Node
        m = record[2]  # Second Node
        relationship = record[1].type  # Relationship type

        # Use element_id as the unique identifier for nodes
        n_id = n.element_id
        m_id = m.element_id

        # Add nodes with their labels and roles as attributes
        G.add_node(n_id, label=n['name'], role=n['role'])
        G.add_node(m_id, label=m['name'], role=m['role'])

        # Add edges with relationship types as labels
        G.add_edge(n_id, m_id, label=relationship)

    return G

def visualize_graph(graph_database: GraphDatabase):
    graph = create_graph(graph_database)
    net = Network(notebook=True, directed=True)  # Directed graph visualization

    # Add nodes to pyvis network with role displayed in tooltip
    for node, attrs in graph.nodes(data=True):
        net.add_node(
            node, 
            label=attrs.get('label'), 
            title=f"Role: {attrs.get('role')}", 
            color="lightblue", 
            shape="dot", 
            size=15
        )

    # Add edges with relationship labels displayed as titles
    for edge in graph.edges(data=True):
        net.add_edge(
            edge[0], 
            edge[1], 
            title=edge[2].get('label'), 
            label=edge[2].get('label')
        )
