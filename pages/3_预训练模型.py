import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chat_models import AzureChatOpenAI

st.set_page_config(page_title="预训练模型", page_icon="📈")
st.markdown("# 预训练模型")
st.sidebar.markdown('''
                    # 预训练模型

在大语言模型问世后，基于文本相似度的问答系统一时间被打入冷宫。通过在大量问题语料上进行训练，语言模型能够在输入一个问题时，输出最符合训练语料和语义的回答。这对于之前是一个巨大的跨越。

我们通过AzureOpenAI进行了简单的实现。            
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

user_question = st.text_input("请输入问题：")

if st.button("向大模型进行提问"):
    st.write(llm.predict(user_question))


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
from langchain.chat_models import AzureChatOpenAI

llm = AzureChatOpenAI(
                        openai_api_base=XXXXX,
                        openai_api_version=XXXXX,
                        deployment_name=XXXXX,
                        temperature=XXXXX,
                        openai_api_key=XXXXX,
                        openai_api_type=XXXXX,
                        streaming=XXXXX)

llm.predict('Python是什么？')

>> 'Python是一种高级编程语言，由Guido van Rossum于1989年开发。它具有简洁、易读、易学的特点，被广泛应用于软件开发、数据分析、人工智能等领域。Python具有丰富的标准库和第三方库，可以用于开发各种类型的应用程序。它支持面向对象编程、函数式编程和过程式编程等多种编程范式。Python的语法简洁明了，代码可读性强，因此被称为“优雅的编程语言”。'
            ''')