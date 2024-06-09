from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader ,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import google.generativeai as Gen_Ai


try:
    Gen_Ai.configure(

        api_key="api_key_here"
    )

except:
    raise Exception("API KEY Error!")

prompt = ("You are a query solver. You are given with sentences; "
               "you need to answer the query by using the given paragraphs ")
model = Gen_Ai.GenerativeModel(model_name='gemini-1.5-pro-latest', system_instruction=prompt)
embed = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
message_Chats = []


db = None

def Load_File(file):
    global db
    try:
        if file.endswith('.docx') or file.endswith('.doc'):
            fileLoad = Docx2txtLoader(file)
        elif file.endswith('.pdf'):
            fileLoad = PyPDFLoader(file)

        loaded = fileLoad.load()
        print(loaded[0].page_content)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=0,
            separators=['\n\n', '\n', ' ', '']
        )

        splits = text_splitter.split_documents(loaded)

        db = Chroma.from_documents(splits,embed)

        return True
    
    except Exception as e:
        print("Error  ",e)
        return False


    
def HFile_Loader(path):
    global db
    try:
        if path.endswith('.docx') or path.endswith('.doc'):
            fileLoad = Docx2txtLoader(path)
        elif path.endswith('.pdf'):
            fileLoad = PyPDFLoader(path)

        loaded = fileLoad.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=0,
            separators=['\n\n', '\n', ' ', '']
        )

        splits = text_splitter.split_documents(loaded)

        db = Chroma.from_documents(splits,embed)

        return True
    
    except Exception as e:
        print("Error",e)
        return False

result = None
def AnswerToQuestion(quest):
    global result
    retrival_chunks = db.similarity_search(quest)
    l = []
    for i in retrival_chunks:
        l.append(i.page_content)
    retrival = retrival = "".join(l)
    print(retrival)
    Whole = f"paragraphs:{retrival}; query:{quest}; generate an answer for the question using the above text of chunks."
    try:
        if message_Chats==[]:
            message_Chats.append({'role':'user','parts':[Whole]})
            result = model.generate_content(message_Chats).text
            return {"complete_query":Whole,"Gen_Airesponse":result}
        else:
            if result!=None:
                message_Chats.append({'role':'model','parts':[result]})
                message_Chats.append({'role':'user','parts':[Whole]})
                result = model.generate_content(message_Chats).text
                return {"complete_query":Whole,"Gen_Airesponse":result}
            else:
                return "Error"
    except Exception as e:
        return "Error: "+str(e)

    