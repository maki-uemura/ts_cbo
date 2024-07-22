from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

# LLM
from langchain.embeddings.openai import OpenAIEmbeddings

# 設定ファイルの読み込み
load_dotenv()

# 定数定義
TMP_FILE_NAME = 'tmp'
TMP_FILE_DIR = os.environ["DB_DIR"]
LLM_PATH = os.environ["MODEL_DIR"] + os.environ["CHATGPT_MODEL_NAME"]
DB_PATH = os.environ["DB_DIR"] + os.environ["CHATGPT_DB_NAME"]


##########################################################
# DB登録
##########################################################
def import_data(uploaded_file):
    if uploaded_file:
        name, ext = os.path.splitext(uploaded_file.name)
        file_path = os.path.join(TMP_FILE_DIR, TMP_FILE_NAME)
        # 一時ファイルとして保存する
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # ファイル形式ごとにローダーを変更
        if ext == '.csv':
            loader = CSVLoader(
                file_path=TMP_FILE_DIR + '/' + TMP_FILE_NAME,
                source_column='source',
                metadata_columns = ['id', 'author'],
                encoding='utf-8'
            )
            
        elif ext == '.xls' or ext == '.xlsx':
            loader = DirectoryLoader(
                TMP_FILE_DIR, 
                glob=TMP_FILE_NAME, 
                loader_cls=UnstructuredExcelLoader, 
                loader_kwargs={'autodetect_encoding': True}
            )
            
        elif ext == '.pdf':
            loader = DirectoryLoader(
                TMP_FILE_DIR, 
                glob=TMP_FILE_NAME, 
                loader_cls=PyPDFLoader, 
            )
            
        else:
            loader = DirectoryLoader(
                TMP_FILE_DIR, 
                glob=TMP_FILE_NAME, 
                loader_cls=TextLoader, 
                loader_kwargs={'autodetect_encoding': True}
            )

        # 入力文字列を分割
        text_splitter = CharacterTextSplitter(
            separator = " ",  
            chunk_size = 1000,  
            chunk_overlap  = 0, 
            length_function=len,
        )
        data = loader.load()
        docs = text_splitter.split_documents(data)

        # LLMに合わせた形式に変換（Embedding）
        embeddings = OpenAIEmbeddings()

        # DB登録
        vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=DB_PATH)
        vectorstore.persist()

        # 一時ファイルを削除
        os.remove(file_path)

        return '完了'

    return '対象なし'
