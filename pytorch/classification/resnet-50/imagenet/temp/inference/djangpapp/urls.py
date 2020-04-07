from django.conf.urls import url
from myapp import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    url(r'^$',views.Userui),
    url(r'^predict$',views.Predict),
    url(r'^inference$',views.Webapp)
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
