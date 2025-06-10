# Python Neo4j Project

This is a sample project to demonstrate how to connect to a Neo4j database using Python.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up your environment variables:**

    Create a `.env` file in the root of the project by copying the `example.env` file:

    ```bash
    cp example.env .env
    ```

    Now, edit the `.env` file with your Neo4j credentials:

    ```
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
    ```

3.  **Run the script:**

    ```bash
    python main.py
    ```

## Jupyter Notebook

To use this project in a Jupyter Notebook, you can install the jupyter kernel and then open the notebook.

1.  **Create a virtual environment** (recommended)

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install the ipykernel**
    ```bash
    python -m ipykernel install --user --name=hc-rag
    ```
4.  **Start Jupyter Notebook**
    ```bash
    jupyter notebook
    ```
    Then you can create a new notebook and use the kernel `hc-rag`.
    You can copy the code from `main.py` into the cells of the notebook.