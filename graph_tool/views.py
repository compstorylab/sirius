from django.shortcuts import render
from django.templatetags.static import static
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

        # If data already exists in the upload directory, display the graph
        if os.path.exists('static/upload_files/graph.json'):
            context['json_file_url'] = static('upload_files/graph.json')

        return context

    def post(self, *args, **kwargs):
        context = {}
        files = self.request.FILES
        form = UploadJSONandImage(self.request.POST, files)

        if os.path.exists('static/upload_files/'):
            shutil.rmtree('static/upload_files/')
        if form.is_valid():
            files_list = UploadJSONandImage.filter(files.getlist('file_field'))
            # form cleaned data only check one file instead a list of files, so add extra validation process above
            for file in files_list:
                upload_file_instance = UploadFileField(file_field=file)
                upload_file_instance.save()
                if file.name == 'graph.json':
                    json_file_url = upload_file_instance.file_field.url
                    context['json_file_url'] = json_file_url

        else:
            form_error = 'File Upload Error'
            context['form_error'] = form_error
        context['upload_form'] = form
        return render(self.request, self.template_name, context)
