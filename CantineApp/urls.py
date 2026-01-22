from django.urls import path
from . import views

app_name = "cantine"

urlpatterns = [
    path("", views.cantine_predict, name="predict"),  # /cantine/
]
