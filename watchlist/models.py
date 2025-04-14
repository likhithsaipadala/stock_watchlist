from django.db import models
from django.contrib.auth.models import User
class Stock(models.Model):
    name = models.CharField(max_length=20)
    user = models.ForeignKey(User, on_delete=models.CASCADE) 
    ticker = models.CharField(max_length=20)
    model_used = models.CharField(max_length=50)
    forecast_days = models.IntegerField()
    features_used = models.TextField()
    predicted_price = models.FloatField()
    mae = models.FloatField()
    rmse = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
         return f"{self.ticker} - {self.model_used} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

