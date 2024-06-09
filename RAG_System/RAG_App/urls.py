from django.urls import path
from . import views

#path('upload/', upload_file, name='upload_file'),

urlpatterns=[
    path('RAG/',views.RAG,name='RAG'),
    path('Upload_File/',views.Upload_File,name='Upload_File'),
    path('SolveQuery/',views.SolveQuery,name='SolveQuery'),
    path('history/<int:chat_id>',views.history,name='history')
    ]