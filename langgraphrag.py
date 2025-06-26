from typing import List, Dict, TypedDict
from langchain_core.pydantic_v1 import Field
from langchain_openai import OpenAIEmbeddings
from langchain_neo4j.graphs import Neo4jGraph
from langchain_neo4j.vectorstores import Neo4jVectorStore
from langgraph.graph import StateGraph, END

# Import the relevance scoring system
from isRelevant import isRelevant, QueryInput, NodeInput, ScorerType, QueryIntent
import numpy as np
from openai import OpenAI
from pydantic import BaseModel as PydanticBaseModel

# --- 1. CONFIGURAZIONE ---
# Configurazione per gemma3:1b tramite Ollama
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:1b"

# Client OpenAI configurato per Ollama
ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_MODEL,
)

# Configurazione Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# Inizializza Embeddings e connessione al Grafo
embeddings_model = OpenAIEmbeddings()
graph = Neo4jGraph(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)


# Funzione helper per chiamate LLM
def call_ollama_llm(system_prompt: str, user_prompt: str, response_format=None):
    """Helper function per chiamare gemma3:1b tramite Ollama."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if response_format:
        response = ollama_client.beta.chat.completions.parse(
            model=OLLAMA_MODEL,
            messages=messages,
            response_format=response_format,
        )
        return response.choices[0].message.parsed
    else:
        response = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
        )
        return response.choices[0].message.content


# --- 2. DEFINIZIONE DELLO STATO DELL'AGENTE ---
class RetrievalState(TypedDict):
    question: str
    query_input: QueryInput  # Structured query for relevance scoring
    documents: List[str]  # Titoli dei documenti trovati
    semantic_results: List[Dict]  # Include metadata for relevance scoring
    graph_query: str
    graph_results: List[Dict]
    candidate_nodes: List[NodeInput]  # Nodi candidati per la rilevanza
    relevant_nodes: List[NodeInput]  # Nodi filtrati per rilevanza
    expanded_subgraph: List[Dict]  # Sottografo espanso
    decision: str
    final_answer: str
    revision_history: List[str]  # Per tenere traccia dei tentativi


# --- 3. DEFINIZIONE DEI NODI DEL GRAFO DI ESECUZIONE ---


def analyze_query(state: RetrievalState) -> Dict:
    """Nodo per analizzare la query e creare la struttura QueryInput."""
    print("--- 🔍 NODO: Analisi Query ---")
    question = state["question"]

    query_input = create_query_input(question)
    print(f"Intent rilevato: {query_input.intent.value}")
    print(f"Entità estratte: {query_input.entities}")

    return {"query_input": query_input}


def retrieve_semantic(state: RetrievalState) -> Dict:
    """Nodo per la ricerca vettoriale iniziale."""
    print("--- 🔍 NODO: Ricerca Semantica ---")
    question = state["question"]

    # Inizializza il retriever vettoriale
    vector_store = Neo4jVectorStore(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        embedding=embeddings_model,
        index_name="chunk_embeddings",
        node_label="Chunk",
        text_node_property="testo",
        embedding_node_property="embedding",
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )  # Aumentiamo k per più candidati

    # Esegui la ricerca
    results = retriever.invoke(question)

    # Estrai i titoli dei documenti per l'espansione
    documents = []
    semantic_results = []

    for doc in results:
        # Aggiungi metadati per la valutazione di rilevanza
        result_data = {
            "testo": doc.page_content,
            "metadata": doc.metadata,
            "id": doc.metadata.get("id", ""),
            "tipo": "chunk",
        }
        semantic_results.append(result_data)

        # Trova il documento padre
        res = graph.query(
            f"MATCH (c:Chunk {{id: '{doc.metadata['id']}'}})<-[:HA_CHUNK]-(d:Documento) RETURN d.titolo as titolo"
        )
        if res and res[0]["titolo"] not in documents:
            documents.append(res[0]["titolo"])

    return {"semantic_results": semantic_results, "documents": documents}


def generate_graph_query(state: RetrievalState) -> Dict:
    """Nodo per generare la query Cypher per espandere il contesto."""
    print("--- 🧠 NODO: Generazione Query Grafo ---")

    query_input = state["query_input"]
    documents = state["documents"]

    # Personalizza la query in base all'intent
    if query_input.intent == QueryIntent.PRODUCT_SEARCH:
        query_focus = "prodotti e loro specifiche"
    elif query_input.intent == QueryIntent.DOCUMENT_REQUEST:
        query_focus = "documenti e manuali"
    elif query_input.intent == QueryIntent.TECHNICAL_SUPPORT:
        query_focus = "informazioni tecniche e troubleshooting"
    else:
        query_focus = "informazioni correlate"

    system_prompt = """Sei un esperto di Neo4j. Data una domanda, le entità estratte e dei documenti di partenza, 
genera una singola query Cypher per trovare informazioni contestuali rilevanti focalizzate su uno specifico focus.

Schema del Grafo:
Nodi: Documento(titolo: string), Chunk(testo: string), Entita(nome: string, tipo: string)
Relazioni: (Documento)-[:HA_CHUNK]->(Chunk), (Chunk)-[:MENZIONA]->(Entita)

Cerca connessioni strutturali che possano fornire contesto aggiuntivo per il focus specificato.
Restituisci solo la query Cypher, nient'altro."""

    user_prompt = f"""Domanda: {state["question"]}
Entità estratte: {query_input.entities}
Documenti di partenza: {documents}
Focus della ricerca: {query_focus}

Genera la query Cypher:"""

    try:
        query = call_ollama_llm(system_prompt, user_prompt)
        return {"graph_query": query.strip()}
    except Exception as e:
        print(f"⚠️ Errore nella generazione della query: {e}")
        # Query di fallback
        fallback_query = "MATCH (d:Documento)-[:HA_CHUNK]->(c:Chunk) RETURN d.titolo, c.testo LIMIT 5"
        return {"graph_query": fallback_query}


def execute_graph_query(state: RetrievalState) -> Dict:
    """Nodo per eseguire la query sul grafo."""
    print(f"--- ⚡ NODO: Esecuzione Query Grafo ---\nQuery: {state['graph_query']}")
    try:
        results = graph.query(state["graph_query"])
        return {"graph_results": results if results else []}
    except Exception as e:
        print(f"Errore query: {e}")
        return {"graph_results": []}


def create_candidate_nodes(state: RetrievalState) -> Dict:
    """Nodo per creare i nodi candidati dalla ricerca semantica e dal grafo."""
    print("--- 🎯 NODO: Creazione Nodi Candidati ---")

    candidate_nodes = []

    # Aggiungi nodi dalla ricerca semantica
    for result in state["semantic_results"]:
        node = create_node_input_from_result(result, "chunk")
        candidate_nodes.append(node)

    # Aggiungi nodi dalla ricerca sul grafo
    for result in state["graph_results"]:
        node = create_node_input_from_result(result, "graph_result")
        candidate_nodes.append(node)

    print(f"Creati {len(candidate_nodes)} nodi candidati")
    return {"candidate_nodes": candidate_nodes}


def score_relevance(state: RetrievalState) -> Dict:
    """Nodo per valutare la rilevanza dei nodi candidati."""
    print("--- 📊 NODO: Scoring Rilevanza ---")

    query_input = state["query_input"]
    candidate_nodes = state["candidate_nodes"]

    # Valuta ogni nodo candidato
    scored_nodes = []
    for node in candidate_nodes:
        score = isRelevant(query_input, node, ScorerType.COMPOSITE)
        node.score = score  # Aggiungi il punteggio al nodo
        scored_nodes.append(node)
        print(f"Nodo: {node.text[:50]}... - Score: {score:.3f}")

    # Ordina per rilevanza e prendi i top N
    relevant_nodes = sorted(scored_nodes, key=lambda x: x.score, reverse=True)[:5]

    print(f"Selezionati {len(relevant_nodes)} nodi rilevanti")
    return {"relevant_nodes": relevant_nodes}


def expand_subgraph(state: RetrievalState) -> Dict:
    """Nodo per espandere il sottografo intorno ai nodi rilevanti."""
    print("--- 🕸️ NODO: Espansione Sottografo ---")

    relevant_nodes = state["relevant_nodes"]
    expanded_subgraph = []

    # Per ogni nodo rilevante, trova le sue connessioni
    for node in relevant_nodes:
        # Estrai ID o identificatori dal nodo
        node_id = node.graph_relations.get("id", "")

        if node_id:
            # Query per espandere le relazioni
            expansion_query = f"""
            MATCH (n {{id: '{node_id}'}})-[r]-(connected)
            RETURN n, type(r) as relation_type, connected
            LIMIT 3
            """

            try:
                expansion_results = graph.query(expansion_query)
                expanded_subgraph.extend(expansion_results)
            except Exception as e:
                print(f"Errore nell'espansione per nodo {node_id}: {e}")

    print(f"Sottografo espanso con {len(expanded_subgraph)} connessioni")
    return {"expanded_subgraph": expanded_subgraph}


def evaluate_context(state: RetrievalState) -> Dict:
    """Nodo per valutare se il contesto raccolto è sufficiente."""
    print("--- 🤔 NODO: Valutazione Contesto ---")

    class Decision(PydanticBaseModel):
        """La decisione se il contesto è sufficiente o necessita di revisione."""

        decision: str = Field(description="'sufficiente' o 'revisione'")
        reasoning: str = Field(description="Breve spiegazione della decisione")

    relevant_nodes = state["relevant_nodes"]
    query_input = state["query_input"]

    # Verifica se abbiamo abbastanza nodi rilevanti con score alto
    high_relevance_nodes = [n for n in relevant_nodes if getattr(n, "score", 0) > 0.7]

    context_summary = f"""
    Nodi rilevanti totali: {len(relevant_nodes)}
    Nodi ad alta rilevanza (>0.7): {len(high_relevance_nodes)}
    Intent della query: {query_input.intent.value}
    """

    top_nodes_text = "\n".join(
        [
            f"- {node.text[:100]}... (score: {getattr(node, 'score', 0):.3f})"
            for node in relevant_nodes[:3]
        ]
    )

    system_prompt = """Sei un supervisore di un sistema RAG. Valuta se il contesto raccolto è sufficiente per 
rispondere alla domanda dell'utente.

Se il contesto sembra completo e rilevante per l'intent, rispondi 'sufficiente'.
Se il contesto è scarso o irrilevante, rispondi 'revisione'."""

    user_prompt = f"""Domanda: {state["question"]}
Intent rilevato: {query_input.intent.value}
Cronologia tentativi: {state.get("revision_history", [])}

Analisi del contesto:
{context_summary}

Top 3 nodi rilevanti:
{top_nodes_text}

Valuta se il contesto è sufficiente per rispondere alla domanda."""

    try:
        decision = call_ollama_llm(system_prompt, user_prompt, Decision)
        print(f"Decisione: {decision.decision} - {decision.reasoning}")
        return {"decision": decision.decision}
    except Exception as e:
        print(f"⚠️ Errore nella valutazione del contesto: {e}")
        # Fallback: se abbiamo nodi ad alta rilevanza, consideriamo sufficiente
        if len(high_relevance_nodes) >= 2:
            print(
                "🔧 Fallback: contesto considerato sufficiente (>= 2 nodi ad alta rilevanza)"
            )
            return {"decision": "sufficiente"}
        else:
            print("🔧 Fallback: contesto considerato insufficiente")
            return {"decision": "revisione"}


def revise_question(state: RetrievalState) -> Dict:
    """Nodo per riformulare la domanda se il contesto è insufficiente."""
    print("--- ✍️ NODO: Revisione Domanda ---")

    current_query = state["query_input"]

    system_prompt = """Sei un esperto di ricerca. La domanda precedente non ha prodotto risultati sufficienti.
Riformula la domanda per un approccio diverso.

Suggerimenti:
- Usa sinonimi diversi
- Cambia il focus della ricerca
- Sii più o meno specifico
- Considera un intent diverso

Genera solo la nuova domanda riformulata."""

    user_prompt = f"""Domanda Originale: {state["question"]}
Intent corrente: {current_query.intent.value}
Entità trovate: {current_query.entities}
Cronologia tentativi: {state.get("revision_history", [])}

Riformula la domanda per ottenere risultati migliori:"""

    try:
        new_question = call_ollama_llm(system_prompt, user_prompt)

        # Aggiorna la cronologia
        history = state.get("revision_history", [])
        history.append(state["question"])

        print(f"📝 Nuova domanda: {new_question.strip()}")
        return {"question": new_question.strip(), "revision_history": history}

    except Exception as e:
        print(f"⚠️ Errore nella revisione della domanda: {e}")
        # Fallback: piccola modifica alla domanda originale
        original = state["question"]
        fallback_question = f"Cerca informazioni su: {original}"

        history = state.get("revision_history", [])
        history.append(state["question"])

        print(f"🔧 Fallback domanda: {fallback_question}")
        return {"question": fallback_question, "revision_history": history}


def generate_answer(state: RetrievalState) -> Dict:
    """Nodo per generare la risposta finale."""
    print("--- 💬 NODO: Generazione Risposta Finale ---")

    relevant_nodes = state["relevant_nodes"]
    query_input = state["query_input"]

    # Prepara il contesto dai nodi rilevanti
    context_text = "\n\n".join(
        [
            f"Fonte {i + 1} (rilevanza: {getattr(node, 'score', 0):.3f}): {node.text}"
            for i, node in enumerate(relevant_nodes)
        ]
    )

    system_prompt = f"""Sei un assistente AI specializzato in {query_input.intent.value}. Rispondi alla domanda dell'utente 
basandoti sul contesto fornito, che è stato selezionato per rilevanza.

Istruzioni:
1. Usa solo le informazioni dal contesto fornito
2. Indica il livello di confidenza nella risposta
3. Se il contesto è insufficiente, sii onesto a riguardo
4. Struttura la risposta in modo chiaro e utile"""

    user_prompt = f"""Domanda: {state["question"]}
Intent: {query_input.intent.value}
Entità rilevanti: {", ".join(query_input.entities)}

--- CONTESTO RILEVANTE ---
{context_text}
--- FINE CONTESTO ---

Fornisci una risposta completa e accurata basata sul contesto:"""

    try:
        answer = call_ollama_llm(system_prompt, user_prompt)
        return {"final_answer": answer.strip()}

    except Exception as e:
        print(f"⚠️ Errore nella generazione della risposta: {e}")
        # Fallback: risposta basica
        fallback_answer = f"""Mi dispiace, ho riscontrato un errore nella generazione della risposta. 

Contesto disponibile:
{context_text[:500]}...

Per la domanda '{state["question"]}', suggerisco di riformulare la richiesta o contattare il supporto tecnico."""

        return {"final_answer": fallback_answer}


# --- 3. FUNZIONI HELPER PER IL SISTEMA DI RILEVANZA ---


class QueryIntentResponse(PydanticBaseModel):
    """Risposta strutturata per l'analisi dell'intent della query."""

    intent: str = Field(
        description="L'intent della query: product_search, document_request, technical_support, comparison_request, o specification_inquiry"
    )
    confidence: float = Field(
        description="Livello di confidenza nell'intent rilevato (0-1)"
    )
    reasoning: str = Field(
        description="Breve spiegazione del perché è stato scelto questo intent"
    )


def analyze_query_intent(question: str) -> QueryIntent:
    """Analizza l'intent della query dell'utente tramite LLM."""

    system_prompt = """Sei un esperto di analisi delle intenzioni degli utenti. Il tuo compito è classificare le domande degli utenti in una delle seguenti categorie:

1. **product_search**: L'utente cerca prodotti specifici, spesso con criteri come prezzo, colore, caratteristiche
   - Esempi: "Find red mountain bikes under $1000", "Cerca biciclette da montagna economiche"

2. **document_request**: L'utente vuole documenti, manuali, guide, istruzioni
   - Esempi: "Show me the manual", "Voglio la documentazione", "Dove trovo le istruzioni?"

3. **technical_support**: L'utente ha problemi tecnici, cerca aiuto, troubleshooting
   - Esempi: "My bike is broken", "Come risolvo questo problema?", "La bici non funziona"

4. **comparison_request**: L'utente vuole confrontare prodotti o opzioni
   - Esempi: "Compare bike A vs bike B", "Differenze tra X e Y", "Quale è migliore?"

5. **specification_inquiry**: L'utente cerca specifiche tecniche, caratteristiche dettagliate
   - Esempi: "Technical specs of X", "Caratteristiche tecniche", "Specifiche del prodotto"

Analizza la domanda e restituisci l'intent più appropriato con confidenza e ragionamento."""

    user_prompt = f"Analizza questa domanda e determina l'intent: '{question}'"

    try:
        response = call_ollama_llm(system_prompt, user_prompt, QueryIntentResponse)

        # Converti la stringa intent in enum
        intent_mapping = {
            "product_search": QueryIntent.PRODUCT_SEARCH,
            "document_request": QueryIntent.DOCUMENT_REQUEST,
            "technical_support": QueryIntent.TECHNICAL_SUPPORT,
            "comparison_request": QueryIntent.COMPARISON_REQUEST,
            "specification_inquiry": QueryIntent.SPECIFICATION_INQUIRY,
        }

        intent_str = response.intent.lower()
        detected_intent = intent_mapping.get(intent_str, QueryIntent.PRODUCT_SEARCH)

        print(
            f"🤖 Intent rilevato: {detected_intent.value} (confidenza: {response.confidence:.2f})"
        )
        print(f"📝 Ragionamento: {response.reasoning}")

        return detected_intent

    except Exception as e:
        print(f"⚠️ Errore nell'analisi dell'intent: {e}")
        print("🔧 Usando fallback: PRODUCT_SEARCH")
        return QueryIntent.PRODUCT_SEARCH


def extract_entities_from_query(question: str) -> List[str]:
    """Estrae entità dalla query (versione semplificata)."""
    # Implementazione semplificata - in produzione usare NER più sofisticato
    import re

    # Estrai parole chiave comuni per prodotti
    entities = []

    # Colori
    colors = re.findall(
        r"\b(rosso|blu|verde|nero|bianco|red|blue|green|black|white)\b",
        question.lower(),
    )
    entities.extend(colors)

    # Tipi di prodotto
    products = re.findall(
        r"\b(mountain bike|bici|bicicletta|manubrio|handlebar|freno|brake)\b",
        question.lower(),
    )
    entities.extend(products)

    # Rimuovi duplicati e restituisci
    return list(set(entities))


def create_query_input(question: str) -> QueryInput:
    """Crea un QueryInput strutturato dalla domanda dell'utente."""
    intent = analyze_query_intent(question)
    entities = extract_entities_from_query(question)

    # Genera embeddings per la query (mock per ora)
    embeddings = np.random.rand(384)  # In produzione, usa embeddings_model

    return QueryInput(
        text=question, embeddings=embeddings, entities=entities, intent=intent
    )


def create_node_input_from_result(
    result: Dict, result_type: str = "unknown"
) -> NodeInput:
    """Converte un risultato del grafo in un NodeInput per la valutazione di rilevanza."""

    # Estrai il testo dal risultato
    text = ""
    if "testo" in result:
        text = result["testo"]
    elif "titolo" in result:
        text = result["titolo"]
    elif "nome" in result:
        text = result["nome"]
    else:
        text = str(result)

    # Determina il tipo di nodo
    node_type = result_type
    if "tipo" in result:
        node_type = result["tipo"]
    elif "label" in result:
        node_type = result["label"].lower()

    # Estrai entità (versione semplificata)
    entities = extract_entities_from_query(text)

    # Genera embeddings (mock per ora)
    embeddings = np.random.rand(384)

    # Relazioni del grafo
    graph_relations = {
        k: v for k, v in result.items() if k not in ["testo", "embeddings"]
    }

    node = NodeInput(
        text=text,
        embeddings=embeddings,
        graph_relations=graph_relations,
        node_type=node_type,
        entities=entities,
    )

    # Aggiungi attributo score (inizialmente 0)
    node.score = 0.0

    return node


# --- 4. COSTRUZIONE DEL GRAFO DI ESECUZIONE CON LANGGRAPH ---

workflow = StateGraph(RetrievalState)

# Aggiungi i nodi
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("retrieve_semantic", retrieve_semantic)
workflow.add_node("generate_graph_query", generate_graph_query)
workflow.add_node("execute_graph_query", execute_graph_query)
workflow.add_node("create_candidate_nodes", create_candidate_nodes)
workflow.add_node("score_relevance", score_relevance)
workflow.add_node("expand_subgraph", expand_subgraph)
workflow.add_node("evaluate_context", evaluate_context)
workflow.add_node("revise_question", revise_question)
workflow.add_node("generate_answer", generate_answer)

# Definisci la logica degli archi (edges)
workflow.set_entry_point("analyze_query")
workflow.add_edge("analyze_query", "retrieve_semantic")
workflow.add_edge("retrieve_semantic", "generate_graph_query")
workflow.add_edge("generate_graph_query", "execute_graph_query")
workflow.add_edge("execute_graph_query", "create_candidate_nodes")
workflow.add_edge("create_candidate_nodes", "score_relevance")
workflow.add_edge("score_relevance", "expand_subgraph")
workflow.add_edge("expand_subgraph", "evaluate_context")

# Arco condizionale: dopo la valutazione, o si genera la risposta o si rivede la domanda
workflow.add_conditional_edges(
    "evaluate_context",
    lambda x: x["decision"],
    {
        "sufficiente": "generate_answer",
        "revisione": "revise_question",
    },
)

# Il ciclo: dopo aver rivisto la domanda, si torna all'analisi (che creerà un nuovo QueryInput)
workflow.add_edge("revise_question", "analyze_query")
workflow.add_edge("generate_answer", END)

# Compila il grafo
app = workflow.compile()


# --- 6. ESECUZIONE DELL'AGENTE CON RILEVANZA INTEGRATA ---

if __name__ == "__main__":
    # Esempi di domande con diversi intent
    domande_esempio = [
        "Find red mountain bikes under $1000",
        "Qual è la relazione tra il Progetto Titano e il Progetto Phoenix?",
        "Show me the technical specifications for mountain bike handlebars",
        "I need the maintenance manual for my mountain bike",
    ]

    # Usa la prima domanda come esempio
    domanda_utente = domande_esempio[0]

    inputs = {
        "question": domanda_utente,
        "revision_history": [],
        "documents": [],
        "semantic_results": [],
        "graph_results": [],
        "candidate_nodes": [],
        "relevant_nodes": [],
        "expanded_subgraph": [],
    }

    print(f"🚀 Avvio del sistema RAG avanzato con rilevanza integrata")
    print(f"📝 Domanda: '{domanda_utente}'\n")
    print("=" * 80)

    try:
        final_state = None
        step_count = 0

        # Usiamo lo stream per vedere l'agente "pensare" passo dopo passo
        for event in app.stream(inputs, {"recursion_limit": 8}):
            if "__end__" not in event:
                step_count += 1
                node_name = list(event.keys())[0]
                print(
                    f"\n--- STEP {step_count}: {node_name.upper().replace('_', ' ')} ---"
                )

                # Mostra informazioni specifiche per ogni nodo
                node_data = event[node_name]

                if node_name == "analyze_query" and "query_input" in node_data:
                    qi = node_data["query_input"]
                    print(f"🎯 Intent: {qi.intent.value}")
                    print(f"🏷️  Entità: {qi.entities}")

                elif node_name == "score_relevance" and "relevant_nodes" in node_data:
                    nodes = node_data["relevant_nodes"]
                    print(f"📊 Top {len(nodes)} nodi rilevanti:")
                    for i, node in enumerate(nodes[:3], 1):
                        score = getattr(node, "score", 0)
                        print(f"   {i}. {node.text[:60]}... (score: {score:.3f})")

                elif node_name == "evaluate_context" and "decision" in node_data:
                    print(f"🤔 Decisione: {node_data['decision']}")

                print("-" * 40)
            else:
                final_state = event

        print("\n" + "=" * 80)
        print("✅ ELABORAZIONE COMPLETATA!")
        print("=" * 80)

        if final_state and "generate_answer" in final_state:
            print("\n🎯 RISPOSTA FINALE:")
            print("-" * 40)
            print(final_state["generate_answer"]["final_answer"])

            # Mostra anche statistiche finali se disponibili
            if "relevant_nodes" in final_state["generate_answer"]:
                relevant_nodes = final_state["generate_answer"]["relevant_nodes"]
                print(f"\n📈 STATISTICHE:")
                print(f"   • Nodi rilevanti utilizzati: {len(relevant_nodes)}")
                avg_score = (
                    sum(getattr(n, "score", 0) for n in relevant_nodes)
                    / len(relevant_nodes)
                    if relevant_nodes
                    else 0
                )
                print(f"   • Score medio di rilevanza: {avg_score:.3f}")
        else:
            print("❌ Errore: impossibile generare una risposta finale")

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("🔚 Fine elaborazione")
    print("=" * 80)
