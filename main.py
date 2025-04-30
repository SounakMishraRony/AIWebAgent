import asyncio
import os
import streamlit as st
from textwrap import dedent

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

# Set up page layout
st.set_page_config(page_title="AI Power Browser", layout="wide", page_icon="🌐")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main-header {
            font-size: 48px;
            font-weight: 800;
            text-align: center;
            color: #FF5722; /* Vibrant Orange */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        }
        .sub-header {
            font-size: 20px;
            text-align: center;
            color: #666666; /* Grey */
            margin-bottom: 20px;
        }
        .sidebar-header {
            font-size: 22px;
            color: #4CAF50; /* Green */
            font-weight: bold;
            margin-bottom: 10px;
        }
        .query-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
        }
        .result-box {
            background-color: #F9F9F9;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #DDDDDD;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<h1 class='main-header'>🌐 AI Power Browser</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Your intelligent web browsing assistant—navigate, interact, and complete tasks effortlessly.</p>",
    unsafe_allow_html=True,
)

# Sidebar with instructions and commands
with st.sidebar:
    st.markdown("<h3 class='sidebar-header'>Quick Commands</h3>", unsafe_allow_html=True)

    st.markdown("#### **Navigation**")
    st.markdown("- `Go to wikipedia.org/wiki/computer_vision`")
    st.markdown("- `Search for the top news in AI today`")

    st.markdown("#### **Interaction**")
    st.markdown("- `Click on the first link on the page`")
    st.markdown("- `Scroll to the bottom and summarize the content`")
    st.markdown("- `Take a screenshot of the main section`")

    st.markdown("#### **Multi-step Tasks**")
    st.markdown("- `Navigate to wikipedia.org/wiki/computer_vision, click on 'Object Detection', and summarize`")
    st.markdown("- `Scroll down and extract all bullet points on the page`")
    st.markdown("- `Search for a Python tutorial, read the first result, and provide a summary`")

    st.markdown("#### **Custom Actions**")
    st.markdown("- `Log in to a demo website and navigate to the dashboard`")
    st.markdown("- `Fill out a form and submit it automatically`")
    st.markdown("- `Extract all links and organize them as a list`")

    st.caption("This AI agent is powered by Puppeteer and OpenAI, delivering unparalleled browsing intelligence.")

# Text area for user commands
query = st.text_area(
    "Your Command",
    placeholder="Example: 'Go to wikipedia.org and summarize the article on computer vision.'",
    height=100,
    label_visibility="collapsed",
    help="Enter commands for navigation, interactions, or multi-step tasks.",
    key="query_input",
)

# Initialize session state variables
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mcp_app = MCPApp(name="streamlit_mcp_agent")
    st.session_state.mcp_context = None
    st.session_state.mcp_agent_app = None
    st.session_state.browser_agent = None
    st.session_state.llm = None
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

# Asynchronous function to set up the agent
async def setup_agent():
    if not st.session_state.initialized:
        try:
            # Start MCP context and application
            st.session_state.mcp_context = st.session_state.mcp_app.run()
            st.session_state.mcp_agent_app = await st.session_state.mcp_context.__aenter__()

            # Configure the browser agent with Puppeteer
            st.session_state.browser_agent = Agent(
                name="browser",
                instruction="""You are a helpful web browsing assistant that can interact with websites using puppeteer.
                - Navigate websites and perform browser actions (click, scroll, type).
                - Extract information from web pages.
                - Take screenshots of page elements when useful.
                - Provide concise summaries of web content using markdown.
                - Follow multi-step browsing sequences to complete tasks.
                """,
                server_names=["puppeteer"],
                servers={
                    "puppeteer": {
                        "name": "puppeteer",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
                    }
                },
            )

            # Initialize the agent and attach LLM
            await st.session_state.browser_agent.initialize()
            st.session_state.llm = await st.session_state.browser_agent.attach_llm(OpenAIAugmentedLLM)

            # Log tools and mark as initialized
            logger = st.session_state.mcp_agent_app.logger
            tools = await st.session_state.browser_agent.list_tools()
            logger.info("Tools available:", data=tools)

            st.session_state.initialized = True
        except Exception as e:
            return f"Error during initialization: {str(e)}"
    return None

# Asynchronous function to run the MCP agent
async def run_mcp_agent(message):
    if not os.getenv("OPENAI_API_KEY"):
        return "Error: No API Key"

    try:
        error = await setup_agent()
        if error:
            return error

        # Generate a result using the LLM
        result = await st.session_state.llm.generate_str(
            message=message, request_params=RequestParams(use_history=True)
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Run command button
if st.button("Explore", type="primary", use_container_width=True):
    with st.spinner("Processing your request..."):
        result = st.session_state.loop.run_until_complete(run_mcp_agent(query))

    # Display result
    st.markdown("### Response")
    st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)
