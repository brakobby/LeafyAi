from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("about/", views.about, name="about"),
    path("predict/upload/", views.predict_upload, name="predict_upload"),
    path("predict/webcam/", views.predict_webcam, name="predict_webcam"),
    path("health/", views.health_check, name="health_check"),
]
