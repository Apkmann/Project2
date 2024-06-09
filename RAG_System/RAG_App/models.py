from django.db import models

class ChatRAGMessages(models.Model):
    Chat_Id = models.IntegerField(default=0)
    Title = models.CharField(max_length=50)
    File_Name = models.CharField(max_length=250)
    conversations_For_Ai = models.JSONField(default=list)
    conversations_For_User = models.JSONField(default=list)