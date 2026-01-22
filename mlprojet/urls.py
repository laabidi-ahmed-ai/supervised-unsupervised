from django.contrib import admin
from django.urls import path, include
from PublicpriveApp.views import home
from PublicpriveApp import views as public_views
from YassinApp import views as viewsyassin
from PerformanceApp import views as performance_views
from RiskApp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('cantine/', include('CantineApp.urls')),
    path('formation/', include('FormationApp.urls')), 
    path('', public_views.home, name='home'),
    path('', include('PublicpriveApp.urls')), 
    path('ml/', include('BoardinPlusIfRulesApp.urls')),
    path('ObjectiveYassin2/', viewsyassin.objective_yassin2, name='objective_yassin2'),
    path('ObjectiveYassin1/', viewsyassin.objective_yassin1, name='objective_yassin1'),
    path('', include('PublicpriveApp.urls')),    # inclut /predict/

    path('api/extract_csv_data/', viewsyassin.extract_csv_data, name='extract_csv_data'),
    path('api/run_prediction/', viewsyassin.run_prediction, name='run_prediction'),
    path('api/extract_csv_data_yassin2/', viewsyassin.extract_csv_data_yassin2, name='extract_csv_data_yassin2'),
    path('api/run_prediction_yassin2/', viewsyassin.run_prediction_yassin2, name='run_prediction_yassin2'),
    path('temporal-trajectory-clustering/', performance_views.temporal_trajectory_clustering, name='temporal_trajectory_clustering'),
    path('school-ranking-by-score/', performance_views.school_ranking_by_score, name='school_ranking_by_score'),
    path('ferdws_cluster/',include('ferdws_cluster.urls')),
    path('ferdws_type/', include('ferdws_type.urls')),
    path('risk/',views.predict_sport_risk,name='risk'),
]
