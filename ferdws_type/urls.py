from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_view, name='predict'),  # l'URL finale sera /ferdws_type/predict/
]