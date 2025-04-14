from django.urls import path
from . import views
from .views import home, delete_stock, stock_api, predict_stock, logout_view
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('', home, name='home'),
    path('delete/<int:stock_id>/', delete_stock, name='delete_stock'),
    path('api/stocks/', stock_api, name='stock_api'),
    path('predict/', predict_stock, name='predict_stock'),
    
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('signup/', views.signup, name='signup'),
    path('logout/', logout_view, name='logout'),
    path('debug/', views.debug_prediction_view, name='debug'),
    path('history/', views.history_view, name='history'),
    path('delete-history/', views.delete_history, name='delete_history'),

]
