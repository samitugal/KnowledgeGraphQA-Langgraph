# Use the official Neo4j image as the base
FROM neo4j:latest

# Set environment variables (optional)
ENV NEO4J_AUTH=neo4j/password123

# Expose the necessary ports
EXPOSE 7474 7473 7687

# Set the data directory (optional)
# VOLUME /data

# Set the default command to run when starting the container
CMD ["neo4j"]

