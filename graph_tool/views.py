from django.shortcuts import render
from django.templatetags.static import static
from django.http import HttpResponse
from django.views.generic import TemplateView
from .forms import UploadJSONandImage
from .models import UploadFileField
import shutil
import os
from sirius_graph_tool import settings


class HomePageView(TemplateView):
    template_name = 'base.html'

    def get_context_data(self, *args, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        form = UploadJSONandImage()
        context['upload_form'] = form
        context['allow_upload'] = not settings.USE_OUTPUT_FOLDER

        if settings.USE_OUTPUT_FOLDER:
            output_prefix = settings.STATICFILES_DIRS[1][0]
            output_dir = settings.STATICFILES_DIRS[1][1]
            if os.path.exists(os.path.join(output_dir, 'graph.json')):
                context['json_file_url'] = static(output_prefix + '/graph.json')
            context['chart_png_path'] = f'{static("")}{output_prefix}/charts/'
            context['chart_json_path'] = f'{static("")}{output_prefix}/json/'
        else:
            # If data already exists in the upload directory, display the graph
            if os.path.exists('static/upload_files/graph.json'):
                context['json_file_url'] = static('upload_files/graph.json')
            context['chart_png_path'] = f'{static("")}upload_files/'
            context['chart_json_path'] = f'{static("")}upload_files/'

        return context

    def post(self, *args, **kwargs):
        context = self.get_context_data(*args, **kwargs)
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
