from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about', views.about, name='about'),
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('custom_logout', views.custom_logout, name='custom_logout'),
    path('home', views.home, name='home'),
    path('view', views.view, name='view'),
    path('model', views.model, name='model'),
    path('predict', views.predict, name='predict'),
]