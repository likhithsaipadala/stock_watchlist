from django import forms
from .models import Stock

class StockForm(forms.ModelForm):
    class Meta:
        model = Stock
        fields = ['name']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. RELIANCE.NS'})
        }

