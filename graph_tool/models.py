from django.db import models

# Create your models here.
class UploadFileField(models.Model):
    file_field = models.FileField(upload_to='static/upload_files/')