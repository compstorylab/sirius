from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
from .forms import UploadJSONandImage
from .models import UploadFileField
import shutil
import os


class HomePageView(TemplateView):
    template_name = 'base.html'

    def get_context_data(self, *args, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        form = UploadJSONandImage()
        context['upload_form'] = form

        return context

    def post(self, *args, **kwargs):
        context = {}
        files = self.request.FILES
        form = UploadJSONandImage(self.request.POST, files)
        if os.path.exists('static/upload_files/'):
            shutil.rmtree('static/upload_files/')
        if form.is_valid() and UploadJSONandImage.validate_extension(files.getlist('file_field')):
            # form cleaned data only check one file instead a list of files, so add extra validation process above
            for file in files.getlist('file_field'):
                upload_file_instance = UploadFileField(file_field=file)
                upload_file_instance.save()
                if UploadJSONandImage.get_extension(file.name) == 'json':
                    json_file_url = upload_file_instance.file_field.url
                    context['json_file_url'] = json_file_url

        else:
            form_error = 'File Upload Error'
            context['form_error'] = form_error
        context['upload_form'] = form
        return render(self.request, self.template_name, context)
