from django.urls import path
from selection_model.views import f1_score_api, index

urlpatterns = [
    path('', index, name='index'),
    path('f1_score/', f1_score_api, name='f1_score_api'),
]
