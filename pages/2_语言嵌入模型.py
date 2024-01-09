import streamlit as st
from text2vec import SentenceModel, cos_sim, semantic_search


st.set_page_config(page_title="文本嵌入模型", page_icon="📈")
st.markdown("# 文本嵌入模型")
st.sidebar.markdown('''
                    # 文本嵌入模型
前面提到的TF-IDF使用了基于统计的文本嵌入方式，后续随着NLP的发展，又出现了语言模型。这时候已经能通过语言模型将文本嵌入为更高维的输入了，一定程度上也能够理解语义。

我们使用Text2Vec进行了实现，它核心使用了transformers库的text2vec-base-chinese嵌入模型。             
                    ''')

embedder = SentenceModel()

# 允许用户上传文件
question_file = st.file_uploader("请选择问题语料", type=["txt"])
if question_file is not None:
    question_data = question_file.getvalue().decode("utf-8")
    question_list = question_data.splitlines()
    
    st.text("转换后的问题列表：")
    st.write(question_list)
    corpus_embeddings = embedder.encode(question_list)

answer_file = st.file_uploader("请选择答案语料", type=["txt"])
if answer_file is not None:
    answer_data = answer_file.getvalue().decode("utf-8")
    answer_list = answer_data.splitlines()
    
    st.text("转换后的答案列表：")
    st.write(answer_list)

top_k = st.number_input("选择 top_k 的值:", min_value=1, max_value=5, value=3, step=1)

if st.button("使用嵌入语言模型对每个问题进行遍历"):
    progress_bar = st.progress(0)
    for index, query in enumerate(question_list):
        progress_bar.progress((index + 1) / len(question_list))

        query_embedding = embedder.encode(query)

        hits = semantic_search(query_embedding, corpus_embeddings, top_k=top_k )

        st.markdown("---")
        st.markdown(f"**Query:** {query}")
        st.markdown(f"\n\n**语料中最相似的{top_k}个回答：**")
        hits = hits[0]  
        for hit in hits:
            st.markdown(f"- {question_list[hit['corpus_id']]}  (Score: **{hit['score']:.4f}**)")

st.markdown(''' 
---      
## 特点
            
在一定程度上，这种语言模型只是对传统基于统计的嵌入方式进行了改良，本质检索上并没有脱离相似度的桎梏。当然，后续Bert等语言模型能够进行文本分类，文本续写等任务，但是在问答任务上仍然不是一个很好的解决方案。
''')

st.markdown('''
---           
我们附上源码如下：
```python
from text2vec import SentenceModel, cos_sim, semantic_search

# 使用了transformers库的text2vec-base-chinese嵌入模型
embedder = SentenceModel()

# 语料样本库
corpus = [
    '花呗更改绑定银行卡',
    '我什么时候开通了花呗',
    'A man is eating food.',
    'A man is eating a piece of bread.',
    'The girl is carrying a baby.',
    'A man is riding a horse.',
    'A woman is playing violin.',
    'Two men pushed carts through the woods.',
    'A man is riding a white horse on an enclosed ground.',
    'A monkey is playing drums.',
    'A cheetah is running behind its prey.'
]

# 将语料进行嵌入
corpus_embeddings = embedder.encode(corpus)
corpus_embeddings
            

queries = [
    '如何更换花呗绑定银行卡',
    'A man is eating pasta.',
    'Someone in a gorilla costume is playing a set of drums.',
    'A cheetah chases prey on across a field.']
            

for query in queries:
    query_embedding = embedder.encode(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=3)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\n语料中最相似的三个回答：")
    hits = hits[0]  
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
            ''')