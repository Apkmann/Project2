from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from RAG_App.RAG_Processor import Load_File,db, AnswerToQuestion, HFile_Loader, message_Chats
from .models import ChatRAGMessages
import os


current_chat_id , is_file_processed = 0,False

@csrf_exempt
def RAG(request):
    #ChatRAGMessages.objects.all().delete()
    global current_chat_id
    if db!=None:
        doc_ids = db.get()['ids']

        if doc_ids:
            db.delete(doc_ids)  
    message_Chats.clear()

    recent_chat = ChatRAGMessages.objects.last() 
    current_chat_id = recent_chat.Chat_Id + 1 if recent_chat else current_chat_id + 1
    all_chats = ChatRAGMessages.objects.all()

    temp_path = "RAG_App//temp_folder//"
    temp_files = os.listdir(temp_path)
    if temp_files:
        os.remove(temp_path + temp_files[0])  

    return render(request, 'RAGApplication.html', {'chatids': all_chats})


@csrf_exempt
def Upload_File(request):
    global current_chat_id, is_file_processed
    if request.method == "POST":
        uploaded_file = request.FILES['file']

        with open(f"RAG_App//temp_folder//{uploaded_file.name}", 'wb+') as TempFile:  
                for chunk in uploaded_file.chunks():
                    TempFile.write(chunk)

        is_file_processed = Load_File(f"RAG_App//temp_folder//{uploaded_file.name}")  
        print(is_file_processed)
        if is_file_processed:
            with open(f"RAG_App//FileStore//{uploaded_file.name}", 'wb+') as media_file:  
                for chunk in uploaded_file.chunks():
                    media_file.write(chunk)

            ChatRAGMessages.objects.create(Chat_Id=current_chat_id, File_Name=f"RAG_App//FileStore//{uploaded_file.name}", Title=uploaded_file.name)
            return JsonResponse({"response": f"File: {uploaded_file.name} Uploaded Successfully!"}) 
        else:
            return JsonResponse({"response": "Failed to Upload"})
    else:
        return JsonResponse({"response": "Failed. Please give proper request"})


@csrf_exempt
def SolveQuery(request):
    global current_chat_id
    if request.method == "POST":
        user_query = request.POST.get('query', None)

        if user_query:
            answer_data = AnswerToQuestion(user_query)
            try:
                if "Error" not in answer_data and is_file_processed:
                    RAGchat = ChatRAGMessages.objects.get(Chat_Id=current_chat_id)
                
                    RAGchat.conversations_For_Ai.append({'role':'user','parts':[answer_data["complete_query"]]})
                    RAGchat.conversations_For_Ai.append({'role':'model','parts':[answer_data["Gen_Airesponse"]]})

                    
                    RAGchat.conversations_For_User.append({'role':'user','parts':[user_query]})
                    RAGchat.conversations_For_User.append({'role':'model','parts':[answer_data["Gen_Airesponse"]]}) 
                    RAGchat.save()

                    return JsonResponse({"answer": answer_data["Gen_Airesponse"]})
                else:
                    return JsonResponse({"answer": answer_data})

            except Exception as e:
                return JsonResponse({"answer": "Error: " + str(e)})
        else:
            return JsonResponse({"answer": "Error"})
    else:
        return JsonResponse({"answer": "Error Wrong request"})


@csrf_exempt
def history(request, chat_id):
    global current_chat_id, is_file_processed
    if db!=None:
        doc_ids = db.get()['ids']

        if doc_ids:
            db.delete(doc_ids)
    message_Chats.clear()

    current_chat_id = chat_id  
    previous_chat = ChatRAGMessages.objects.get(Chat_Id=chat_id)
    is_file_processed = HFile_Loader(previous_chat.File_Name)

    if is_file_processed:

        for  i in previous_chat.conversations_For_Ai:
            message_Chats.append(i)

        return JsonResponse({"history": previous_chat.conversations_For_User, "filename": previous_chat.Title})
    else:
        return JsonResponse({"history": "Failed"})