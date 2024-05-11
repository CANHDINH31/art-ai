from django.urls import path, include
from . import views

urlpatterns = [
	path('build-model/', views.buildModel, name="buildModel"),
    path('prediction/', views.prediction, name="prediction"),
] 