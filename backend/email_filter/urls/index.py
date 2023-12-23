from django.urls import path, include
from email_filter.views.index import solve

urlpatterns = [
    path('submit/', solve),
]