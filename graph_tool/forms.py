from django import forms

class UploadJSONandImage(forms.Form):
    file_field = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True})
    )

    @classmethod
    def validate_extension(cls, files_list):
        """files_list: a list of uploadedFile obj"""
        all_passed = True
        for f in files_list:
            extension = cls.get_extension(f.name)
            all_passed = all_passed and (extension in ['json', 'png'])
        return all_passed

    @classmethod
    def filter(cls, files_list):
        """files_list: a list of uploadedFile obj"""
        return filter(lambda f: cls.get_extension(f.name) in ['json', 'png'], files_list)

    @staticmethod
    def get_extension(file_name):
        if file_name is None:
            return ''
        elif len(file_name) == 0:
            return ''
        else:
            return file_name[file_name.rfind('.') + 1:].lower()

