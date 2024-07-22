from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from my_library import my_print

################################
# データ読込
# https://ninthcode.net/230725-vectorstore/
################################
def select_data_csv(query="株式会社ニコニコファミリー", interface = None):
    my_print('CSVデータ取得の開始', interface, add_time = True)

    top_k = 2

    embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
    db = Chroma(embedding_function=embeddings, persist_directory="./my_chroma")
    docs = db.similarity_search_with_score(query, k=top_k)
    print(db._collection.count())

    for i in range(top_k):
        get_data = docs[i][0].metadata['id'] + " " + docs[i][0].metadata['author'] + " " + docs[i][0].page_content + " score:" + str(docs[i][1])
        my_print('取得したもの：' + get_data)

    # 終了日時
    my_print('CSVデータ取得の完了', interface, add_time = True)

def select_data_txt(query, interface = None):
    my_print('TXTデータ取得の開始', interface, add_time = True)

    top_k = 2

    embeddings = LlamaCppEmbeddings(model_path="./llama-2-7b-chat.Q2_K.gguf", n_ctx=4096)
    db = Chroma(embedding_function=embeddings, persist_directory="./my_chroma")
    docs = db.similarity_search_with_score(query, k=top_k)
    print(db._collection.count())

    for i in range(top_k):
        get_data = docs[i]
        my_print('取得したもの：' + str(get_data))

    # 終了日時
    my_print('TXTデータ取得の完了', interface, add_time = True)

