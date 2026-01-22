from django.urls import path
from .views import predict_etablissement, home,cluster_summary_view,predict_cluster_view,dashboard_view

app_name = "PublicpriveApp"  # IMPORTANT

urlpatterns = [
    path("", home, name="home"),
    path("predict/", predict_etablissement, name="predict_etablissement"),
    path('clusters/', cluster_summary_view, name='cluster_summary'),
    path('predict-cluster/', predict_cluster_view, name='predict_cluster'),
    path('dashboard/', dashboard_view, name='dashboard'),



]