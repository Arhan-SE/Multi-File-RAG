
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Cohere
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.document_loaders import PyPDFLoader
import os
from PIL import Image
import google.generativeai as genai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_community.document_loaders import WebBaseLoader
import sqlite3

llm = Cohere()

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata or {}

def truncate_text(text, max_length):
    return text[:max_length]

def get_gemini_response_for_txt(input,txt,prompt):
  response=model.generate_content([input,txt,prompt])
  return response.text


def text_rag(file_content, query):
    # Truncate file content if too long
    max_chunk_size = 4000  # Set a safe limit considering prompt + document
    truncated_content = truncate_text(file_content, max_chunk_size)

    documents = [Document(truncated_content)]
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = CohereEmbeddings(user_agent="app")

    store = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings , store, namespace=embeddings.model
    )

    db = FAISS.from_documents(docs, cached_embedder)

    template = ChatPromptTemplate.from_template("""
        You are a helpful assistant. You will be uploaded with a txt file in which some questions might be asked.
        Answer the question based on the relevant documents. If the answer is not based on the relevant documents,
        then just say i dont know or say what you know.
    """)

    # Ensure prompt length is within limits


    relevant_docs=db.similarity_search(query)

    relevant_docs_content = " ".join([doc.page_content for doc in relevant_docs])

    formatted_template=template.format(query=query)


    res = get_gemini_response_for_txt(formatted_template,relevant_docs_content,query)

    return res


def get_gemini_response_for_pdf(input,pdf,prompt):
  response=model.generate_content([input,pdf,prompt])
  return response.text

def pdf_reader(file_path, query):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    embeddings = CohereEmbeddings(user_agent="app")
    store = LocalFileStore("./cache/")

    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )

    db = FAISS.from_documents(docs, cached_embedder)

    template = ChatPromptTemplate.from_template("""
        You are a helpful assistant. You will be uploaded with a PDF file in which some questions might be asked.
        Answer the question based on the relevant documents. If the answer is not based on the relevant documents,
        then just say 'I don't know' or say what you know.
    """)

    relevant_docs=relevant_docs=db.similarity_search(query)

    relevant_docs_content = " ".join([doc.page_content for doc in relevant_docs])

    formatted_template=template.format(query=query)
    res = get_gemini_response_for_pdf(formatted_template,relevant_docs_content,query)
    res = res.replace("\\n", "\n")
    return res


genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

#function to load gemini pro
model=genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response_for_image(input,image,prompt):
  response=model.generate_content([input,image[0],prompt])
  return response.text


def input_image_details(uploaded_file):
  if uploaded_file is not None:
    bytes_data=uploaded_file.getvalue()

    image_parts=[
        {
            "data":bytes_data,
            "mime_type":uploaded_file.type
        }
    ]
    return image_parts
  else:
    return FileNotFoundError("File not found")

def get_gemini_response_for_csv(input,relevant_doc,prompt):
  model=genai.GenerativeModel("gemini-pro")
  response=model.generate_content([input])
  return response.text

def csv_reader(file,query):
  loader=CSVLoader(file_path=file)
  documents=loader.load()

  text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
  docs=text_splitter.split_documents(documents)

  embeddings=CohereEmbeddings(user_agent="app")

  store = LocalFileStore("./cache/")

  cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )

  db=FAISS.from_documents(docs,cached_embedder)
  relevant_docs = db.similarity_search(query)

    # Extract the content from the relevant documents
  relevant_docs_content = " ".join([doc.page_content for doc in relevant_docs])

  prompt=ChatPromptTemplate.from_template("""You are an expert CSV reader.
   A human will upload a CSV, and you must answer based on it, accurately referencing rows and columns.
   Keep the format clean with proper spaces and new lines. If unsure, just say what you know without making up answers.
   Use exact data from the file without repeating or replacing it.
   Be friendly, and in conversations, avoid checking the documents unless necessary
   These are the relevant documents <content> {relevant_docs} <content> question: {query}""")

  formatted_prompt = prompt.format(relevant_docs=relevant_docs_content, query=query)
  #res=llm.invoke(prompt.format(relevant_docs=db.similarity_search(query),query=query)).replace("\n",' ')
  res=get_gemini_response_for_csv(formatted_prompt,relevant_docs_content, query)

  res = res.replace("\\n", "\n")

  return res


def get_gemini_response_for_link(input,link,prompt):
  response=model.generate_content([input,link,prompt])
  return response.text

def link_reader(link,query):
  loader = WebBaseLoader(link)

  document = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  docs = text_splitter.split_documents(document)

  embeddings = CohereEmbeddings(user_agent="app")

  store = LocalFileStore("./cache/")

  cached_embedder = CacheBackedEmbeddings.from_bytes_store(
      embeddings, store, namespace=embeddings.model
  )

  db = FAISS.from_documents(docs, cached_embedder)

  relevant_docs = db.similarity_search(query)

  # Extract the content from the relevant documents
  relevant_docs_content = " ".join([doc.page_content for doc in relevant_docs])

  prompt = ChatPromptTemplate.from_template(
      """You are an expert assistant. Based on the provided documents, answer the user's query in a clean and structured paragraph. Present the information in a readable way, using sentences and avoiding '\n' or any line breaks.

        If there are multiple points or items, list them in sentence form using commas or semi-colons to separate them.

        Ensure the response is concise, factual, and based strictly on the documents provided. If the user asks a casual question (e.g., "Hello" or "How are you?"), respond in a friendly, conversational tone. Avoid adding irrelevant information and maintain accuracy from the relevant documents.
        Related document: <content>related_docs_content<content> question: {query}
        """
  )
  formatted_prompt = prompt.format(relevant_docs=relevant_docs_content, query=query)
  res = get_gemini_response_for_link(formatted_prompt,relevant_docs_content, query)

  res = res.replace("\\n", "\n")

  return res

def get_gimini_response_for_db(query,prompt):
  model=genai.GenerativeModel('gemini-pro')
  response=model.generate_content([prompt[0], query])
  return response.text

def get_gimini_response_for_nice_db(prompt_2,query,res):
  model=genai.GenerativeModel('gemini-pro')
  response=model.generate_content([prompt_2, query,res])
  return response.text

def read_sql_query(query, db):
  connection=sqlite3.connect(db)
  cursor=connection.cursor()
  cursor.execute(query)
  result=cursor.fetchall()
  connection.close()

  for row in result:
    print(row)

  return row


prompt_for_db=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS,
    SECTION AND MARKS\n\nFor example,\nExample 1 - How many entries of records are present?,
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENT ;
    \nExample 2 - Tell me all the students studying in Data Science class?,
    the SQL command will be something like this SELECT * FROM STUDENT
    where CLASS="Data Science";
    also the sql code should not have ``` in beginning or end and sql word in output
    Dont give any rubbish answers. Make sure the answer's are precise and correct.
    """
]

prompt_for_nice_db="""
You are an expert writer.
Once, you had a brother who was an expert in English questions to SQL query.
You got seperated by some consequences and now you meet each other after 10 years.
Your brother tells that he only has the talent to give the direct answer but not in a sentence.
That's why he asks you to help him make a sentence with the output that he is giving.
Cherished that you have met him after 10 years you do it very affectively and properly.

Also while answering dont state anything about your brother or your past just the answer.

This is the query and the result, Make a sentence for the output.

"""



# Streamlit app
st.set_page_config(page_title="CHATOOOO")

st.title("CHATOOOO")

# Track the selected file type
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'query' not in st.session_state:
    st.session_state.query = None

file_type = st.selectbox("Choose file type to upload:", ('Text', 'PDF', 'Image', 'CSV', 'LINK','DB', 'SQLITE'))

# Reset query when file type changes
if file_type != st.session_state.file_type:
    st.session_state.file_type = file_type
    st.session_state.query = None  # Clear the query

uploaded_file = None
link_input = None

if file_type in ['Text', 'PDF', 'Image', 'CSV','DB' ,'SQLITE']:
    uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf', 'jpg', 'png', 'jpeg', 'csv','db', 'sqlite'])
elif file_type == 'LINK':
    link_input = st.text_input("Enter the link:")

# Update query state
query = st.text_input("Enter your query:", key="query")

if st.session_state.query:
    if file_type in ['Text', 'PDF', 'CSV','DB' ,'SQLITE']:
        if uploaded_file:
            file_name = f"temp_{file_type.lower()}_file.txt"
            with open(file_name, "wb") as file:
                file.write(uploaded_file.getbuffer())

    if file_type == 'Text' and uploaded_file is not None:
        st.write("Processing text file...")
        file_content = uploaded_file.read().decode("utf-8")
        result = text_rag(file_content, st.session_state.query)
        st.subheader("Response: ")
        st.write(result)

    elif file_type == 'PDF':
        st.write("Processing PDF file...")
        result = pdf_reader(file_name, st.session_state.query)
        st.subheader("Response: ")
        st.write(result)

    elif file_type == 'Image':
        input_prompt = "You are an expert in telling what's in the image like the features and many more things in a very unique style. The human will upload and ask some questions based on the image and you have to answer those questions only based on the image uploaded."
        image_data = input_image_details(uploaded_file)
        response = get_gemini_response_for_image(input_prompt, image_data, st.session_state.query)
        st.subheader("Response: ")
        st.write(response)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    elif file_type == 'CSV':
        st.write("Processing CSV file...")
        result = csv_reader(file_name, st.session_state.query)
        st.subheader("Response: ")
        st.write(result)

    elif file_type == 'LINK' and link_input:
        st.write(f"Processing link: {link_input}")
        result = link_reader(link_input, st.session_state.query)
        st.subheader("Response: ")
        st.write(result)

    elif file_type == 'DB':
        st.write("Processing DB/SQLITE file...")
        response=get_gimini_response_for_db(st.session_state.query,prompt_for_db)
        data=read_sql_query(response,file_name)

        st.subheader("Response")

        for row in data:
          result_for_text=get_gimini_response_for_nice_db(prompt_for_nice_db,st.session_state.query,row)
          st.write(result_for_text)

    elif file_type == 'SQLITE':
      st.write("Processing SQLITE file...")
      response=get_gimini_response_for_db(st.session_state.query,prompt_for_db)
      data=read_sql_query(response,file_name)

      st.subheader("Response")

      for row in data:
        result_for_text=get_gimini_response_for_nice_db(prompt_for_nice_db,st.session_state.query,row)
        st.write(result_for_text)
