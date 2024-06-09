import chromadb #importing chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from langchain_community.document_loaders import Docx2txtLoader ,PyPDFLoader #importing document loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter #importing text splitters
import os
from dotenv import load_dotenv,find_dotenv
import google.generativeai as ai #For Ai Model
from celery import shared_task
_ = load_dotenv(find_dotenv())

if _:
    ai.configure(
        api_key=os.environ["api_key"]
    )

else:
    raise Exception("API KEY Not Found!")

instruction = ("You are a query solver. You are provided sentences; "
               "you need to answer the query by using the given sentences "
               "and your response must be in HTML tags without indicating it's HTML.")
model = ai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=instruction)

messages = []
embedding_func = GoogleGenerativeAiEmbeddingFunction(api_key=os.environ["api_key"])

settings = Settings(
    persist_directory="db"
)


client = chromadb.Client(settings)
collection = client.get_or_create_collection(name="file",embedding_function=embedding_func)

global_collection = collection
def FileLoader(file):
    try:
        if file.endswith('.docx') or file.endswith('.doc'):
            fileLoad = Docx2txtLoader(file)
        elif file.endswith('.pdf'):
            fileLoad = PyPDFLoader(file)
        else:
            raise ValueError("Wrong File Format")

        loaded = fileLoad.load()
        content = [word.page_content.strip() for word in loaded if word]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            separators=['\n\n', '\n', '. ', ' ', '']
        )

        splitted_document = text_splitter.split_text("\n\n".join(content))

        ids = [str(Id) for Id in range(len(splitted_document))]
        documents = [doc for doc in splitted_document]

        Is_ok = load_c.apply_async(ids, documents)
        return Is_ok.get()
    except Exception as e:
        print('Error:', e)
        return False

@shared_task
def load_c(Id, document):
    try:
        global_collection.add(
        ids=Id,
        documents=document
        )
        return True  
    except:
        return False
    
def History_File_Loader(path):
    #For retriving the file and again going to the same process above
    try:
        if path.endswith('.docx') or path.endswith('.doc'):
            fileLoad = Docx2txtLoader(path)
        else:
            fileLoad = PyPDFLoader(path)
        loaded = fileLoad.load()
        content = [word.page_content.strip() for word in loaded if word]
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #size of chunks 
        chunk_overlap=0, #overlaping of chunks
        separators = ['\n\n','\n','. ',' ',''] #seperators
        )
        
        splitted_document = text_splitter.split_text("\n\n".join(content))
        print(splitted_document[0])

        ids = [str(Id) for Id in range(len(splitted_document))]
        documents = [doc for doc in splitted_document]
        collection.add(
                ids=ids,
                documents = documents
            )

        return True
    
    except Exception as e:
        return False

def expand_query(query):
    try:
        instruction = ("You are a query generator. You are provided with query also correct the grammer or spelling if any mistakes in query; "
                   "you need to generate 3 queries by using the given query and your response must be a normal text the question seperated by ';' like 'q1; q2; q3' and don't include any acknowledgements only pure queries not any other words")
        model = ai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction=instruction)
        queries =  model.generate_content(f"Query: {query}").text
        print(queries)
        return queries
    except Exception as e:
        print("Error:",e)
        return 'Error'


result = ""
def Generate_answer(query):
    global result
    #It will give the matching chunks for the query
    queries = expand_query(query)
    if 'Error' not in queries:
        paragraph = collection.query(
            query_texts = [queries],
            n_results=3 
            )
    else:
        return "<h3 style='color:red;'>An Error Occured</h3>"
    print(paragraph)
    sentences = "".join(paragraph['documents'][0])
    complete_query = f"sentences:{sentences}; query:{query}; generate an answer for the query using the above sentences."
    try:
        if messages==[]:
            messages.append({'role':'user','parts':[complete_query]})
            result = model.generate_content(messages).text
            return {"complete_query":complete_query,"Airesponse":result}
        else:
            if result!=None:
                messages.append({'role':'model','parts':[result]})
                messages.append({'role':'user','parts':[complete_query]})
                result = model.generate_content(messages).text
                return {"complete_query":complete_query,"Airesponse":result} #It will return reponses from Ai and chunks with query
            else:
                return "An Error Occured: "
    except Exception as e:
        return "<h3 style='color:red;'>An Error Occured: "+str(e)+"</h3>"

    