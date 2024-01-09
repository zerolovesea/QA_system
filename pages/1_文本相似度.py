import io
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



st.set_page_config(page_title="基于统计方法的QA系统", page_icon="📈")
st.markdown("# 基于统计方法的QA系统")
st.sidebar.markdown('''
                    # 基于统计方法的QA系统
我们排开早期的按照规则实现的问答系统，最早被我们了解到的QA系统应该是通过文本嵌入+文本相似度实现的。

首先文本嵌入有多种实现方式，这里大致介绍一下：

文本嵌入是将文本转换为数值向量的过程，使得可以在这些向量之间进行相似性计算。TF-IDF就是一个常见的实现方式：

TF-IDF（Term Frequency-Inverse Document Frequency）：
TF-IDF 是一种统计方法，用于评估一个词对于一个文档集或一个单独的文档的重要性。它基于词频（TF）和逆文档频率（IDF）的乘积来为每个词赋予权重。               
                    ''')



# 允许用户上传文件
question_file = st.file_uploader("请选择问题语料", type=["txt"])

if question_file is not None:
    question_data = question_file.getvalue().decode("utf-8")
    # st.text("问题语料：")
    # st.write(question_data)
    # print(question_data)

    question_list = question_data.splitlines()
    
    st.text("转换后的问题列表：")
    st.write(question_list)

answer_file = st.file_uploader("请选择答案语料", type=["txt"])

if answer_file is not None:
    answer_data = answer_file.getvalue().decode("utf-8")
    answer_list = answer_data.splitlines()
    
    st.text("转换后的答案列表：")
    st.write(answer_list)


if st.button("使用TF-IDF对问答对进行向量化"):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(question_list + answer_list)
    tfidf_matrix

    st.text("可以看到问答对被转化为了向量，这时候就可以进行相似度搜索了")

# 允许用户输入问题
user_question = st.text_input("请输入问题：")

if st.button("返回最相似的问题"):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(question_list + answer_list)
    user_question_vec = vectorizer.transform([user_question])

    similarities = cosine_similarity(user_question_vec, tfidf_matrix)[0]

    most_similar_idx = np.argmax(similarities)
    st.markdown("**最相似的问题：**")
    st.write(question_list[most_similar_idx])

    st.markdown("**最相似问题的答案：**")
    st.write(answer_list[most_similar_idx])

st.markdown('''   
---    
## 特点
            
这是最早的QA系统实现手段之一。实现简单易懂，原理也并不复杂。然而它有很多不足，首先，它需要提前准备大量的问答对，以适应不同领域的各个问题，其次对于不同的提问形式，它很难给出精准准确的回答。

本质上，作为基于统计的方法，它本质上并没有理解问题的语义，只是找了一个最像的问答对作为回答。
''')


st.markdown('''
---           
我们附上源码如下：
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 示例的问题和答案数据
questions = [
    "什么是Python?",
    "Python有哪些优点?",
    "如何定义函数?",
    "Python的应用场景是什么?"
]

answers = [
    "Python是一种高级编程语言。",
    "Python有简单易读的语法、丰富的库和广泛的应用场景。",
    "在Python中，函数可以使用def关键字进行定义。",
    "Python在Web开发、数据分析、人工智能等多个领域有广泛的应用。"
]
            
# 使用TF-IDF向量化文本数据
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions + answers)

tfidf_matrix

>> <8x12 sparse matrix of type '<class 'numpy.float64'>'
	with 12 stored elements in Compressed Sparse Row format>            
            
user_question = "Python有哪些应用场景?"
user_question_vec = vectorizer.transform([user_question])

user_question_vec

>> <1x12 sparse matrix of type '<class 'numpy.float64'>'
	with 0 stored elements in Compressed Sparse Row format>            
            
similarities = cosine_similarity(user_question_vec, tfidf_matrix)[0]
most_similar_idx = np.argmax(similarities)

answers[most_similar_idx]

>> 'Python是一种高级编程语言。'
            ''')