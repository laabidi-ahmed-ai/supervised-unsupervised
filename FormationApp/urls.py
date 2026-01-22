from django.urls import path
from . import views

app_name = "formation"

urlpatterns = [
    path("predict/", views.formation_predict, name="predict"),
]
