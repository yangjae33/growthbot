from django import forms
from .models import *


#Video uploading form
class UploadForm(forms.ModelForm):
    class Meta:
        model = File
        fields = '__all__'


#Chatbot information uploading form
class ChatbotUploadForm(forms.ModelForm):
    class Meta:
        model = ChatbotInfo
        fields = '__all__'
