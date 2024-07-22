from dotenv import load_dotenv
import os

# メイン
from llama_cpp import Llama

# 通常用

# RAG構成用
from langchain.llms import LlamaCpp
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
# RAG構成用（LangChainのPromptTemplateを定義）
from langchain.chains.qa_with_sources.map_reduce_prompt import QUESTION_PROMPT
from langchain import PromptTemplate

# 設定ファイルの読み込み
load_dotenv()

# 定数定義
LLM_PATH = os.environ["MODEL_DIR"] + os.environ["LLAMA2_MODEL_NAME"]
DB_PATH = os.environ["DB_DIR"] + os.environ["LLAMA2_DB_NAME"]

################################
# LLMに質問
################################
def query(user_message):

    # AIモデルの読み込み
    llm = Llama(model_path=LLM_PATH)

    # 問い合わせ
    llm_return = llm(
        user_message,
        max_tokens=300,
        echo=False,
    )
    
    # AIモデルの回答を抽出
    answer = llm_return['choices'][0]['text'].rstrip('\r\n')
    
    print('========== デバッグ：回答開始 ==========')
    print(str(answer))
    print('========== デバッグ：回答終了 ==========')
    
    return answer

################################
# LLMに質問（RAG）
################################
def query_with_db(user_message):

    # AIモデルの読み込み
    llm = LlamaCpp(
        model_path=LLM_PATH,
        input={
            "max_tokens": 32,
            "stop": ["System:", "User:", "Assistant:", "\n"],
        },
        verbose=True,
        n_ctx=2048,
    )

    # DBの読み込み
    embeddings = LlamaCppEmbeddings(model_path=LLM_PATH, n_ctx=4096)
    db = Chroma(embedding_function=embeddings, persist_directory=DB_PATH)
    qa =VectorDBQAWithSourcesChain.from_chain_type(llm, chain_type="map_reduce", vectorstore=db)

    # LangChainのAgentとToolsを定義
    tools = [
        Tool(
          name = "my_searcher",
          func=qa,
          description="Langchainの説明"
      )
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # PromptTemplateの定義
    template = """
    Please answer the questions below.
    questions:{question}
    answer：
    """
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )

    question = prompt.format(question=user_message)
    answer = agent.run(question)

    print('========== デバッグ：回答開始 ==========')
    print(str(answer))
    print('========== デバッグ：回答終了 ==========')

    return answer
