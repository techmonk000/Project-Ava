from llama_index.llms.ollama import Ollama 
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,PromptTemplate 
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv 

load_dotenv()

llm = Ollama(model="codellama:13b",request_timeout=160.0)


parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf":parser}
documents = SimpleDirectoryReader("./data", file_extractor= file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

query_engine = vector_index.as_query_engine(llm=llm)

result = query_engine.query("what are some of the routes in api?")
print(result)