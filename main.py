import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek


def _log(msg):
    curr_time = datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
    logging.warning(f"[{curr_time}] {msg}")

def chat(message, histories, bot_tags, gender, api_key=None):
    os.environ["DEEPSEEK_API_KEY"] = api_key
    model = ChatDeepSeek(model="deepseek-chat", temperature=0)

    role = "女朋友" if gender == "男" else "男朋友"
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"你扮演着用户的{role}, 具有如下性格特征：{', '.join(bot_tags)}. 生成 '~' 符号时记得加空格，避免产生错误的删除线。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    conversion_chain = ConversationChain(llm=model, memory=histories, prompt=prompt_template)
    response = conversion_chain.invoke(message)
    return response["response"]

def _required(attr, msg, parent=st):
    if not attr:
        parent.error(msg)
        st.stop()

def main():
    # title
    st.set_page_config(page_title="Soul Mate", page_icon="⭐️", layout="centered")

    # header
    st.header("⭐️ Soul Mate")

    # Left Side Bar
    with st.sidebar:
        api_key = st.text_input("请输入您的API Key", type="password")
        model_name = st.selectbox("请选择模型", ["deepseek-chat"])
        gender = st.selectbox("请选择您的性别", ["男", "女"])
        bot_tags = st.multiselect("请选择TA的性格特征", ["可爱", "温柔", "暴躁", "傲娇", "古灵精怪"], default=["可爱", "古灵精怪"])

    # Body
    # session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [("AI", "你好，有什么可以帮到你")]

    if "histories" not in st.session_state:
        st.session_state["histories"] = ConversationBufferMemory(return_messages=True)

    for name, message in st.session_state["messages"]:
        st.chat_message(name).write(message)

    message = st.chat_input("请输入要发送的消息")
    if message:
        _required(api_key, "您还没有输入API Key，请先输入API Key")

        st.session_state["messages"].append(("human", message))
        st.chat_message("human").write(message)


        with st.spinner("思考中..."):
            if model_name == "deepseek-chat":
                response = chat(message, st.session_state["histories"], bot_tags, gender,api_key)
            else:
                raise NotImplementedError(f"模型 {model_name} 未实现")

        st.session_state["messages"].append(("AI", response))
        st.chat_message("AI").write(response)
        _log(f"[Human] {message}")
        _log(f"   [AI] {response}")

if __name__ == '__main__':
    main()
