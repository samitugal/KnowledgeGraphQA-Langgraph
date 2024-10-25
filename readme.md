# AI-Powered Knowledge Graph and Question Answering System

## Why it is needed

In today's information-rich world, efficiently organizing, understanding, and retrieving knowledge is crucial. This AI-Powered Knowledge Graph and Question Answering System addresses several key challenges:

1. **Information Overload**: With the vast amount of textual data available, it's challenging to quickly grasp the key concepts and relationships within a body of text. This system automatically extracts and visualizes this information as a knowledge graph.

2. **Contextual Understanding**: Traditional keyword-based search often lacks context. By representing information as a graph, this system captures the relationships between entities, enabling more nuanced and context-aware querying.

3. **Flexible Knowledge Base**: The system can continuously expand its knowledge graph with new information, making it adaptable to various domains and evolving knowledge landscapes.

4. **Intelligent Question Answering**: By combining the power of large language models with structured knowledge graphs, the system can provide more accurate and contextually relevant answers to user queries.

5. **Bridging Structured and Unstructured Data**: This system seamlessly integrates unstructured text data with structured graph representations, offering the benefits of both worlds.

6. **Visual Insight**: The graph visualization component allows users to intuitively explore complex relationships within the data, facilitating better understanding and decision-making.

7. **Extensibility**: With its modular architecture, the system can be easily extended to incorporate new data sources, language models, or specialized domain knowledge.

By addressing these needs, this system serves as a powerful tool for researchers, analysts, educators, and anyone dealing with large volumes of textual information, helping them to quickly extract insights, answer questions, and discover new connections within their data.

## Key Features

- Knowledge graph generation from text input
- Question answering based on the generated knowledge graph
- Web search integration for questions not answerable from the graph
- Graph visualization
- Flexible LLM integration (OpenAI and Amazon Bedrock supported)
- Neo4j graph database integration

## Technologies Used

- **Python**: The primary programming language used for the project.
- **LangChain**: For building and managing the AI/LLM pipelines.
- **LangGraph**: Used for creating the workflow graph.
- **Neo4j**: Graph database for storing and querying the knowledge graph.
- **FastAPI**: For creating the API endpoints.
- **Streamlit**: For building the user interface.
- **OpenAI GPT**: Large Language Model for text processing and generation.
- **Amazon Bedrock**: Alternative LLM provider.
- **Tavily Search API**: For web search functionality.
- **NetworkX and Pyvis**: For graph visualization.
- **Docker**: For containerization and easy deployment.

## Project Structure

The project is organized into several key components:

1. **LLM Pipeline**: Handles interactions with language models (OpenAI GPT or Amazon Bedrock).
2. **Graph Database**: Manages the Neo4j database operations.
3. **Content Provider**: Supplies text content for knowledge graph generation.
4. **Workflow Nodes**: Individual processing steps in the application workflow.
5. **API and UI**: FastAPI for backend services and Streamlit for the user interface.

## Key Components

### LLM Pipeline

The LLM pipeline (`src/LLM/Pipeline.py`) is responsible for:
- Generating knowledge graph queries
- Routing user requests
- Answering questions based on graph content
- Generating responses for web search results

### Graph Database

The graph database component (`src/Databases/GraphDatabase.py` and `src/Databases/Neo4j.py`) handles:
- Executing Cypher queries
- Fetching graph data
- Managing database connections

### Workflow

The main workflow (`src/graph.py`) orchestrates the entire process, including:
- Routing user input
- Generating knowledge graphs
- Finding relevant nodes
- Answering questions or performing web searches

Here's a visual representation of the workflow:

![Workflow Graph](graph-schema.png)

### UI

The Streamlit-based UI (`src/ui.py`) provides a user-friendly interface for interacting with the system.

## Configuration

The project uses YAML configuration files for both the LLM and database settings:
- LLM config: `configs/LLM/openai.yaml` or `configs/LLM/bedrock.yaml`
- Database config: `configs/GraphDatabase/neo4j.yaml`

## Setup and Running

### Using Docker (Recommended)

1. Make sure you have Docker and Docker Compose installed on your system.

2. Clone the repository and navigate to the project directory.

3. Create a `.env` file in the root directory and add your environment variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

4. Build and run the containers:
   ```
   docker-compose up --build
   ```

5. Access the Streamlit UI at `http://localhost:8501`

### Manual Setup

1. Install dependencies:
   ```
   poetry install
   ```

2. Set up environment variables in a `.env` file.

3. Run the Streamlit UI:
   ```
   streamlit run src/ui.py
   ```

## Future Improvements

- Implement caching for improved performance
- Add support for more LLM providers
- Enhance the graph visualization capabilities
- Implement user authentication and multi-user support
- Monitoring and logging

## Environment Variables

The project requires certain environment variables to be set in a `.env` file in the root directory. The required variables differ slightly depending on whether you're using OpenAI or AWS (Bedrock) as your LLM provider.

### For OpenAI Configuration:

- `LLM_CONFIG_PATH=configs/LLM/openai.yaml`
- `GRAPH_DATABASE_CONNECTION_PATH=configs/GraphDatabase/neo4j.yaml`
- `OPENAI_API_KEY=your_openai_api_key`
- `TAVILY_API_KEY=your_tavily_api_key`

### For AWS (Bedrock) Configuration:

- `LLM_CONFIG_PATH=configs/LLM/bedrock.yaml`
- `GRAPH_DATABASE_CONNECTION_PATH=configs/GraphDatabase/neo4j.yaml`
- `AWS_ACCESS_KEY_ID=your_aws_access_key_id`
- `AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key`
- `AWS_DEFAULT_REGION=your_aws_region`
- `TAVILY_API_KEY=your_tavily_api_key`

Make sure to replace the placeholder values with your actual API keys and credentials. Keep your `.env` file secure and never commit it to version control.

## Usage

After setting up the project, you can interact with the system through the Streamlit UI or by using the API directly.

### Using the Streamlit UI

1. Start the Streamlit app:
   ```
   streamlit run src/ui.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. You'll see a text input area where you can enter your text or question.

4. For generating a knowledge graph:
   - Enter a paragraph or article of text.
   - Click the "Run" button.
   - The system will generate a knowledge graph and display it visually.

5. For asking questions:
   - Enter your question in the text area.
   - Click the "Run" button.
   - The system will attempt to answer based on the existing knowledge graph.
   - If the answer isn't found in the graph, it will perform a web search.

### Example Workflow

1. Generate a knowledge graph:
   - Input: "John is a student at MIT. He is studying computer science and is passionate about artificial intelligence."
   - The system will create nodes for John, MIT, computer science, and artificial intelligence, with appropriate relationships.

2. Ask a question:
   - Input: "What is John studying?"
   - The system will query the knowledge graph and respond with "John is studying computer science."

3. Ask a question not in the graph:
   - Input: "What is the capital of France?"
   - The system will perform a web search and provide an answer based on the search results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve this project! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

Please ensure your code adheres to the project's coding standards and include tests for new features or bug fixes.

### Reporting Issues

If you encounter any bugs or have suggestions for improvements, please open an issue on the GitHub repository. Provide as much detail as possible, including steps to reproduce the issue and your environment details.

### Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

Thank you for your interest in contributing to our AI-Powered Knowledge Graph and Question Answering System!

### Knowledge Graph Generation

![Knowledge Graph Generation](images/knowledge_graph_generation.png)
*This image shows the process of generating a knowledge graph from input text.*

### Monitoring Utility with Langfuse

The system includes a monitoring utility that leverages Langfuse to track and analyze the performance of the knowledge graph generation and question answering processes. This utility provides real-time insights into system performance, allowing for quick identification and resolution of any issues that may arise.
