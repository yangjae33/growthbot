from django.db import models
from functools import partial

# Create your models here.

def _wrapper(instance, filename, path):
    file_name, ext = filename.split('.')
    return '{}{}.{}'.format(path, file_name, ext)


def update_filename(path):
    return partial(_wrapper, path=path)


#model for uploading video
class File(models.Model):
    file_name = models.CharField(max_length=50, verbose_name="최종 파일명(*)")
    thumbnail = models.ImageField(upload_to=update_filename('thumbnail/'), blank=True, verbose_name="미리보기 이미지")
    video = models.FileField(upload_to=update_filename('video/'), verbose_name="동영상 첨부파일(*)")
    aiml = models.FileField(upload_to=update_filename('aiml/'), blank=True, verbose_name="AIML 첨부파일")


#model for uploading chatbot information
class ChatbotInfo(models.Model):
    video_name = models.CharField(max_length=50, verbose_name="영상명(*)")
    chatbot_name = models.CharField(max_length=50, verbose_name="챗봇명(*)")
    chatbot_ip = models.CharField(max_length=50, verbose_name="챗봇 IP(*)")
    chatbot_port = models.IntegerField(verbose_name="챗봇 PORT (*)")
    chatbot_key = models.CharField(max_length=50, verbose_name="챗봇 키 (*)")
    file_name = models.FileField(upload_to=update_filename('chatbot_data/'), verbose_name="질의 세트 첨부파일(*)")
