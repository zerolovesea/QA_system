import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

st.set_page_config(page_title="基于知识库和大模型的问答系统", page_icon="📈")
st.markdown("# 基于知识库和大模型的问答系统")
st.sidebar.markdown('''
                    # 预训练模型

前面提到基于统计的检索方法并不能以符合语义的方式输出回答，而是调用原文。那么现在我们有了会说人话的大语言模型，能不能将它们结合在一起呢？

基于这个思想，出现了RAG（检索增强生成）。通过结合输入问题和检索到的相关内容，大语言模型将得到的内容进行包装，使得其能够输出合理的，符合语义的回答。

通过Langchain和ChromaDB进行了简单的实现。            
                    ''')


load_dotenv()


llm = AzureChatOpenAI(
                    openai_api_base=os.getenv("OPENAI_API_BASE"),
                    openai_api_version=os.getenv("OPENAI_API_VERSION"),
                    deployment_name=os.getenv("DEPLOYMENT_NAME"),
                    temperature=os.getenv("TEMPERATURE"),
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    openai_api_type=os.getenv("OPENAI_API_TYPE"),
                    streaming=os.getenv("STREAMING")
)
# 允许用户上传文件
question_file = st.file_uploader("请选择语料", type=["txt"])

if question_file is not None:
    try:
        question_data = question_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        try:
            question_data = question_file.getvalue().decode("gbk")
        except:
            st.error("无法解码文件内容。请检查文件编码或使用其他编码尝试。")
    
    print(question_data)
    sentences = question_data.split('。')
    sentences = [s.strip() for s in sentences if s.strip()]

    st.markdown("将会对语料进行切块，并向量化。这可能需要一些时间，请耐心等待。")
    documents = [Document(page_content=s, metadata={'source': 'wiki'}) for s in sentences]

    print(documents)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    st.markdown("语料加载完成。")
    st.markdown("---")
    vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
    retriever = db.as_retriever(search_type="mmr")

    user_question = st.text_input("请输入问题：")
    docs = db.similarity_search(user_question)

    if st.button("执行向量查询"):
        st.markdown('**检索到的最相似的内容：**')
        st.markdown(docs[0].page_content)


    if st.button("向模型进行提问"):
        from langchain.prompts import ChatPromptTemplate

        template = """你是一个用于问答任务的助手。
        使用检索到的上下文来回答问题。
        如果你不知道答案，只需说你不知道。
        最多使用三句话，并保持答案简洁。
        问题：{question}
        上下文：{context}
        答案：
        """
        prompt = ChatPromptTemplate.from_template(template)
        from langchain.schema.runnable import RunnablePassthrough
        from langchain.schema.output_parser import StrOutputParser

        rag_chain = (
            {"context": retriever,  "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        query = user_question
        st.markdown('**模型的回答：**')        
        st.markdown(rag_chain.invoke(query))


st.markdown('''   
            ---    
            ## 特点
                        
            大语言模型经过了上亿问答语料的学习，它能够通过学到过的内容，找到最符合人类逻辑的下一个输出。由于问题是对Python的询问，它从上文中找到了最有可能出现的回答，并以人类能够理解的方式进行输出。

            然而，由于生成语言模型以生成符合语义的句子为目的，因此它无法判断输出的内容是否准确，这也是它目前备受诟病的缺陷之一。
            ''')


st.markdown('''
            ---           
            我们附上源码如下：
            ```python
            from langchain.embeddings import SentenceTransformerEmbeddings
            from langchain.text_splitter import CharacterTextSplitter
            from langchain.vectorstores import Chroma
            from langchain.document_loaders import TextLoader

            # 从本地导入语料
            loader = TextLoader('state_of_the_union.txt')
            documents = loader.load()

            # 将文本进行切块
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)

            # 将语料嵌入后存入向量数据库
            embeddings = SentenceTransformerEmbeddings()
            db = Chroma.from_documents(docs, embeddings)
            
                        
            query = "What did the president say about Ketanji Brown Jackson"
            
            docs = db.similarity_search_with_score(query)
            docs[0]
                        
            from langchain.prompts import ChatPromptTemplate

            template = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.
            Question: {question}
            Context: {context}
            Answer:
            """
            prompt = ChatPromptTemplate.from_template(template)
                        
            from langchain.schema.runnable import RunnablePassthrough
            from langchain.schema.output_parser import StrOutputParser

            rag_chain = (
                {"context": retriever,  "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            query = "What did the president say about Justice Breyer"
            rag_chain.invoke(query)

            vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory='db')
            vectordb.persist()
            vectordb = None
                        
            vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
            retriever = db.as_retriever(search_type="mmr")
            retriever.get_relevant_documents(query)[0]

                        ''')