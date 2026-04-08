from django.urls import path

from .views import health, home, predict_api

urlpatterns = [
    path("", home, name="home"),
    path("health/", health, name="health"),
    path("api/predict/", predict_api, name="predict_api"),
]
