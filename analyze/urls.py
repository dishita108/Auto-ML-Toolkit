from django.contrib import admin
from django.urls import path
from analyze import views


urlpatterns = [
    path('', views.index, name="index"),
    path('result/', views.output, name="output"),
    path('aboutus/', views.aboutus, name="aboutus"),
] 