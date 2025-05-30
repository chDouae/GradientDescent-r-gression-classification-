
from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_file, name='upload'),
    path('preprocess/', views.preprocess, name='preprocess'),
    path('model/', views.model_train, name='model'),
    path('prediction/', views.prediction, name='prediction'),
     path('generate_heatmap/', views.generate_heatmap, name='generate_heatmap'), 
 
]