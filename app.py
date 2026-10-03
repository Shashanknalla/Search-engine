import streamlit as st
from requests.exceptions import RequestException
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool

from dotenv import load_dotenv
load_dotenv()

#Toold Creattion

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)

def search_wikipedia(query: str) -> str:
    try:
        return wiki_wrapper.run(query)
    except RequestException:
        return "Wikipedia returned an invalid response. Try another search source."

wiki = Tool(
    name="Wikipedia",
    func=search_wikipedia,
    description="Search Wikipedia for a concise summary of a topic.",
)

search_wrapper = DuckDuckGoSearchRun()

def search_web(query: str) -> str:
    try:
        return search_wrapper.run(query)
    except DuckDuckGoSearchException:
        return "Web search is temporarily rate-limited by DuckDuckGo. Try Wikipedia or Arxiv, or answer without web results."

search = Tool(
    name="Search",
    func=search_web,
    description="Search the web for current information and useful sources.",
)

st.title("Chat with Search")

"""In this project we are using 'StremlitCallbackHandler' to display the thoughts and actions fo a an agents in interactive wen app"""

st.sidebar.title('Settings')
api_key = st.sidebar.text_input("Enter your Groq API key", type = "password")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [

        {
            "role" : "assistant", "content":" Hello human, I AI am a chatbot who can search the web. Tell me how can I help you " }

    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg["content"])

if prompt:= st.chat_input(placeholder="Type your question here"):
    st.session_state.messages.append({'role':'user', 'content': prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b", streaming=True)
    tools = [search, arxiv, wiki]
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use the available tools when they help answer the user's question."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt_template)
    search_agent = AgentExecutor(agent=agent, tools=tools)

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        result = search_agent.invoke(
            {"input": prompt},
            config={"callbacks": [st_cb]},
        )
        response = result["output"]
        st.session_state.messages.append({'role':'assistant', 'content': response})
        st.write(response)