from dotenv import load_dotenv
import os

# メイン
from langchain.chat_models import ChatOpenAI

# 通常用
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

# RAG構成用
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
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
LLM_PATH = os.environ["CHATGPT_MODEL_NAME"]
DB_PATH = os.environ["DB_DIR"] + os.environ["CHATGPT_DB_NAME"]

################################
# ChatGPTに質問
################################
def query(user_message):

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if OPENAI_API_KEY == '':
        return 'キーが未設定です：ChatGPT'

    #プロンプトテンプレートを作成
    template = """
    あなたは聞かれた質問に答える優秀なアシスタントです。
    """
    # 会話のテンプレートを作成
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    # AIモデルの読み込み
    llm = ChatOpenAI(
        model_name=LLM_PATH,
        temperature=0
    )
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=llm)
        
    # 問い合わせ
    answer = conversation.predict(input=user_message)
    
    return answer

################################
# ChatGPTに質問（RAG）
################################
def query_with_db(user_message):

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if OPENAI_API_KEY == '':
        return 'キーが未設定です：ChatGPT'

    # AIモデルの読み込み
    llm = ChatOpenAI(
        model_name=LLM_PATH,
        temperature=0
    )

    # DBの読み込み
    embeddings = OpenAIEmbeddings()
    db = Chroma(embedding_function=embeddings, persist_directory=DB_PATH)
    qa =VectorDBQAWithSourcesChain.from_chain_type(llm, chain_type="map_reduce", vectorstore=db)

    # LangChainのAgentとToolsを定義
    tools = [
        Tool(
          name = "my_searcher",
          func=qa,
          description="説明：ChatGPTによるRAG"
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
