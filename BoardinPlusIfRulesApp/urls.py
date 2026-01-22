from django.urls import path
from . import views

app_name = 'boardinplusifrules'

urlpatterns = [
    path('boarding/', views.boarding_prediction_view, name='boarding_prediction'),
    path('api/predict-boarding/', views.predict_boarding_api, name='predict_boarding_api'),
    path('association-rules/', views.association_rules_view, name='association_rules'),
    path('api/predict-cluster/', views.predict_cluster_api, name='predict_cluster_api'),
]