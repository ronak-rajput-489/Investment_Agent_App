from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from openai import OpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict
from sqlalchemy import create_engine, text
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import quote_plus
from langchain.embeddings import OpenAIEmbeddings  # Or HuggingFaceEmbeddings
# from dbutils import schema, table_info
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
import io
import base64
import os
load_dotenv()

###########################################################################################################

snowflake_user = os.getenv("SNOWFLAKE_USER")
snowflake_pass = os.getenv("SNOWFLAKE_PASSWORD")
snowflake_acct = os.getenv("SNOWFLAKE_ACCOUNT")  # e.g. ab12345.ap-southeast-1
snowflake_db   = os.getenv("SNOWFLAKE_DATABASE")
snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA")
snowflake_wh = os.getenv("SNOWFLAKE_WAREHOUSE")
snowflake_role = os.getenv("SNOWFLAKE_ROLE")

# 🧠 1. Create a LangGraph-compatible state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Snowflake Engine:
engine = create_engine(
    f"snowflake://{snowflake_user}:{snowflake_pass}@{snowflake_acct}/"
    f"{snowflake_db}/{snowflake_schema}?warehouse={snowflake_wh}&role={snowflake_role}"
)
# 🧠 State
class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]

##########################################################################################################

@tool
def run_raw_sql(query: str) -> str:
    """Run any Snowflake SQL command like CREATE, INSERT, UPDATE, DELETE."""
    try:
        with engine.begin() as conn:
            conn.execute(text(query))
        return "✅ Snowflake SQL command executed successfully."
    except Exception as e:
        return f"❌ Error executing SQL: {e}"

#############################################################################################################

@tool
def run_select_sql(query: str) -> str:
    """Run SELECT queries against the snowflake and return results."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            if not rows:
                return "✅ Query executed, but returned no data."
            return "\n".join(str(row) for row in rows)
    except Exception as e:
        return f"❌ Error executing SQL: {e}"

###############################################################################################################

## RAG Agent
## --> Load PDF
loader = PyPDFLoader("pd_description.pdf")
documents = loader.load()

## --> Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(documents)

## --> Embeddings
embeddings = OpenAIEmbeddings()  # Or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## --> Create Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

## --> Add RAG Tool

retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4.1-nano"),
    retriever=retriever
)

@tool
def rag_lookup(query: str) -> str:
    """Answer questions from knowledge documents using RAG."""
    return rag_chain.run(query)

##################################################################################################################

# Example function to categorize
def categorize_risk(score):
    if score <= 10:
        return 'Low'
    elif 10 < score <= 25:
        return 'Moderate'
    else:
        return 'High'

from sqlalchemy import text

# ------------------- SEGMENT CUSTOMERS -------------------
@tool
def segment_customers(table_name: str) -> str:
    """
    Segments customers into Low, Moderate, High based on their risk_appetite score.
    Args:
        table_name: The name of the database table containing the customer data.
    Returns:
        A message confirming segmentation and showing sample output.
    """
    try:
        query = f"SELECT * FROM {table_name}"

        # ✅ Use SQLAlchemy connection for pandas
        with engine.connect() as conn:
            df_final = pd.read_sql_query(text(query), conn)

        # Apply categorization
        df_final["risk_appetite_class"] = df_final["risk_appetite"].apply(categorize_risk)

        # ✅ Save back using engine.begin()
        with engine.begin() as conn:
            df_final.to_sql(
                name=f"{table_name}_segmented",
                con=conn,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=1000,
            )

        return (
            f"Segmentation done ✅. Saved as {table_name}_segmented. "
            f"Sample:\n{df_final[['customer_id','risk_appetite','risk_appetite_class']].head(5)}"
        )

    except Exception as e:
        return f"❌ Error in segment_customers: {e}"


# ------------------- RECOMMEND PRODUCTS -------------------
@tool
def recommend_products(table_name: str) -> str:
    """
    Recommends investment products for customers based on their risk_appetite_class.
    Args:
        table_name: The name of the database table containing the segmented customer data.
    Returns:
        A message confirming recommendations and showing sample output.
    """
    try:
        query = f"SELECT * FROM {table_name}"

        # ✅ Use SQLAlchemy connection for pandas
        with engine.connect() as conn:
            df_final = pd.read_sql_query(text(query), conn)

        # Ensure segmentation exists
        if "risk_appetite_class" not in df_final.columns:
            raise ValueError("Table must already contain 'risk_appetite_class'. Run segment_customers first.")

        # Map products
        product_mapping = {
            "Low": "A – Investlink",
            "Moderate": "A - Life Wealth Premier",
            "High": "A - Life Infinite"
        }
        df_final["recommended_product"] = df_final["risk_appetite_class"].map(product_mapping)

        new_table = f"{table_name}_with_recommendations"

        # ✅ Save back into Snowflake
        with engine.begin() as conn:
            df_final.to_sql(
                name=new_table,
                con=conn,
                if_exists="replace",
                index=False,
                method="multi",
                chunksize=1000,
            )

        return (
            f"✅ Product recommendations added based on risk appetite. "
            f"Saved as {new_table}. Sample:\n"
            f"{df_final[['customer_id','risk_appetite_class','recommended_product']].head(5)}"
        )

    except Exception as e:
        return f"❌ Error in recommend_products: {e}"

# @tool
# def segment_customers(table_name: str) -> str:
#     """
#     Segments customers into Low, Moderate, High based on their risk_appetite score.
#     Args:
#         table_name: The name of the database table containing the customer data.
#     Returns:
#         A message confirming segmentation and showing sample output.
#     """
#     # 🔹 Load table from schema (example: using SQLAlchemy)
#     #from sqlalchemy import create_engine
#     # engine = create_engine(
#     # f"snowflake://{snowflake_user}:{snowflake_pass}@{snowflake_acct}/"
#     # f"{snowflake_db}/{snowflake_schema}?warehouse={snowflake_wh}&role={snowflake_role}"
#     # )
#     query = f"SELECT * FROM {table_name}"
#     df_final = pd.read_sql(query, engine)

#     # Apply categorization
#     df_final['risk_appetite_class'] = df_final['risk_appetite'].apply(categorize_risk)

#     # Optionally save back to DB
#     df_final.to_sql(table_name + "_segmented", engine, if_exists="replace", index=False)

#     return f"Segmentation done ✅. Saved as {table_name}_segmented. Sample:\n{df_final[['customer_id','risk_appetite','risk_appetite_class']].head(5)}"

# ################################################################################################################

# @tool
# def recommend_products(table_name: str) -> str:
#     """
#     Recommends investment products for customers based on their risk_appetite_class.
#     Args:
#         table_name: The name of the database table containing the segmented customer data.
#     Returns:
#         A message confirming recommendations and showing sample output.
#     """

#     # 🔹 Load table from schema
#     query = f"SELECT * FROM {table_name}"
#     df_final = pd.read_sql(query, engine)

#     # Check if segmentation already exists
#     if 'risk_appetite_class' not in df_final.columns:
#         raise ValueError("Table must already contain 'risk_appetite_class'. Run segment_customers first.")

#     # Mapping dictionary (you can extend this easily)
#     product_mapping = {
#         "Low": "A – Investlink",
#         "Moderate": "A - Life Wealth Premier",
#         "High": "A - Life Infinite"
#     }

#     # Apply mapping
#     df_final['recommended_product'] = df_final['risk_appetite_class'].map(product_mapping)

#     # Save back to DB (new table with recommendations)
#     new_table = table_name + "_with_recommendations"
#     df_final.to_sql(new_table, engine, if_exists="replace", index=False)

#     return (
#         f"✅ Product recommendations added based on risk appetite. "
#         f"Saved as {new_table}. Sample:\n"
#         f"{df_final[['customer_id','risk_appetite_class','recommended_product']].head(5)}"
#     )


#################################################################################################################

tools = [run_raw_sql, run_select_sql, rag_lookup, segment_customers, recommend_products]

#################################################################################################################
    
# Bind tools
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

selected_columns_info = """
Table: CUSTOMER
-   STOCK_EXPOSURE: Amount invested in stocks (RM). Indicates willingness to take market risk.
-   SIP_TYPE: Type of Systematic Investment Plan chosen by the customer, where 0 = Bond (low risk, stable returns), 1 = Equity (high risk, growth-oriented), and 2 = Balanced (medium risk, mix of equity and bonds).
-   FD_HOLDING: Amount held in Fixed Deposits (safe, low-risk). Higher values = conservative mindset.
-   AVG_MONTHLY_LOGINS: Average number of times the customer logs into the platform per month. Digital engagement indicator.
-   CLICKS_ON_EQUITIES: Number of clicks on equity-related products/articles. Shows interest in stocks.
-   DI_RATIO_PCT: Debt-to-Income ratio (EMI/Income). Lower is good (financially healthy), higher means over-leveraged.
-   NET_WORTH: Total assets minus liabilities. Shows overall wealth level.
-   TOTAL_ASSETS: Sum of all assets (FDs, mutual funds, stocks, property, etc.).
-   INCOME: Monthly or annual income of the customer.
-   INVESTMENT_TYPE: category of investment the customer prefers (e.g. Mutual Fund, Fixed Deposit, Bonds, Stocks)

Table: CUSTOMER_segmented
-   STOCK_EXPOSURE: Amount invested in stocks (RM). Indicates willingness to take market risk.
-   SIP_TYPE: Type of Systematic Investment Plan chosen by the customer, where 0 = Bond (low risk, stable returns), 1 = Equity (high risk, growth-oriented), and 2 = Balanced (medium risk, mix of equity and bonds).
-   FD_HOLDING: Amount held in Fixed Deposits (safe, low-risk). Higher values = conservative mindset.
-   AVG_MONTHLY_LOGINS: Average number of times the customer logs into the platform per month. Digital engagement indicator.
-   CLICKS_ON_EQUITIES: Number of clicks on equity-related products/articles. Shows interest in stocks.
-   DI_RATIO_PCT: Debt-to-Income ratio (EMI/Income). Lower is good (financially healthy), higher means over-leveraged.
-   NET_WORTH: Total assets minus liabilities. Shows overall wealth level.
-   TOTAL_ASSETS: Sum of all assets (FDs, mutual funds, stocks, property, etc.).
-   INCOME: Monthly or annual income of the customer.
-   INVESTMENT_TYPE: category of investment the customer prefers (e.g. Mutual Fund, Fixed Deposit, Bonds, Stocks)
-   RISK_APPETITE_CLASS: Customer's risk profile's class
"""

# First model node — decides on tool usage
def model_node(state: AgentState) -> AgentState:
    sys_prompt = SystemMessage(
        content=(
            
            f"""
            You are an AI assistant that helps marketing people to  analyze customer's demographic, financial, invetment data etc. stored in Snowflake. 
            Use {selected_columns_info} as your schema reference.
            
            When the user asks questions like

            "Do all customers behave the same way?" → focus on general behavior patterns (engagement, financial health, and exposure).
            "Are all customers investing in similar products?" → focus on investment type preferences (stable-return vs. growth-seeking).
            "Do we currently segment them in any way?" → focus on segmentation using the risk appetite score (derived from investment, digital, and financial behavior).

            
            When the user asks questions like 
                "Do all customers behave the same way?" 
                    you must:
                        - Compare customers based on the scoring rules (investment, digital, and financial behavior).  
                        - Highlight clear **differences** in customer patterns (e.g., Aggressive Investors, Balanced, Conservative Savers).  
                        - Explain which columns drive these differences (e.g., INCOME, AVG_MONTHLY_LOGINS, NET_WORTH, STOCK_EXPOSURE, SIP_TYPE,INVESTMENT_TYPE, FD_HOLDING, DI_RATIO_PCT).  
                        - Be specific and **always segment customers into groups** (not just general text).
                
                "Are all customers investing in similar products?"
                    you must:
                        - Compare customers based on the scoring rules (investment, digital, and financial behavior).  
                        - Highlight clear **differences** in customer patterns (e.g., Aggressive Investors, Balanced, Conservative Savers).  
                        - Explain which columns drive these differences. 
                        - Only focus on these columns (STOCK_EXPOSURE, INVESTMENT_TYPE, FD_HOLDING, CLICKS_ON_EQUITIES).  
                        - Be specific and **always segment customers into groups** (not just general text).
  

            Example style of answer:
            - "Customers do not behave the same way.  
            - Some are Aggressive Investors (high stock exposure, equity SIPs, frequent logins).  
            - Others are Conservative Savers (prefer FDs, low equity clicks, high DI ratio).  
            - A third group are Balanced Customers (moderate exposure, balanced SIPs, medium logins)."  

            If the user asks about "patterns of product usage" → focus on differences in SIP_TYPE,INVESTMENT_TYPE, STOCK_EXPOSURE, FD_HOLDING and CLICKS_ON_EQUITIES.  
            If the user asks about "risk behavior" → focus on DI_RATIO_PCT, NET_WORTH, INCOME.  
            If the user asks about "segmentation" → explain the **risk appetite score classification**: 
                - Conservative (Low Risk Appetite): Risk Appetite score ≤ 10 → prefer FDs, Bonds, low stock exposure, high DI ratio. 
                - Balanced (Moderate Risk Appetite): Risk Appetite score between 11–25 → mix of Equity + Debt SIPs, moderate FD holdings, average DI ratio. 
                - Aggressive (High Risk Appetite): Risk Appetite score > 25 → equity-focused, high stock allocation, active SIPs, strong engagement. 
                - Ask the human user, Do you want me to segment these customers? 
                    -If the user agrees or say yes, then call the 'segment_customers' tool that applies the following rule on the 'CUSTOMER' data table stored in the schema.   
                                  
            If the user asks about "product suggestion based on customer's risk profile" 
                - Read all available investment products from 'pd_description' documentation. 
    
                
                              
            Always make your response **insightful and actionable**.
        """
        )
    )
    
    # - prefer 'CUSTOMER_segmented' table as a knowledge base.
    # - Suggest products based on the customer’s risk appetite classification (High, Low, High).  
    
    # - Then ask the human user:  
    #                 "Do you want me to recommend specific products for these customers?"  
    #             - If the user say "I would like to recommend products." → **only then call the 'recommend_products' tool** on the 'CUSTOMER_segmented' table.

    # Let the LLM decide next step
    response = model.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]}

###############################################################################################

# Tool execution node — corrected
def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    tool_messages = []

    for call in tool_calls:
        tool_name = call["name"]
        args = call["args"]

        # Find the matching tool
        for t in tools:
            if t.name == tool_name:
                # Run the tool
                result = t.run(args)
                
                # Convert result to string if it's not already
                if isinstance(result, (dict, list)):
                    import json
                    result = json.dumps(result, indent=2)
                    
                # Wrap result in the expected content structure
                tool_messages.append(
                    ToolMessage(
                        content=[{"type": "text", "text": f"Tool {tool_name} executed. Result:\n{result}"}],
                        tool_call_id=call["id"]
                    )
                )

    # Return new state with messages
    return {"messages": tool_messages}

##############################################################################################

# Conditional — do we need another round?
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # Continue only if the last message actually contains a tool call
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", model_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")
app = graph.compile()


############################################################################################### 
