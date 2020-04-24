import os
from threading import Timer
import webbrowser


def start_server():
    """
    Call this function to run the Django server from a Python script.
    It will block the thread until the server is closed. It should
    ran last in the pipeline.

    EX:
    from .server import start_server
    start_server()
    """
    # Start the browser launch timer
    t = Timer(3.0, delay_browser_launch)
    t.start()

    # Setup and run the Django server
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sirius_graph_tool.settings")

    from django.core.management import call_command
    from django.core.wsgi import get_wsgi_application

    # Create the application
    application = get_wsgi_application()

    # Start the development server
    # Turn off reloading when there are changes to the folder structure.
    call_command('runserver',  '127.0.0.1:8000',  '--noreload')


def delay_browser_launch():
    webbrowser.open("http://127.0.0.1:8000")

