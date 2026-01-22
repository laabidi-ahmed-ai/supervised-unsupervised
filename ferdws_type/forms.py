from django import forms

class PredictionForm(forms.Form):
    ULIS = forms.IntegerField(label='ULIS', required=True)
    SEGPA = forms.FloatField(label='SEGPA', required=True)
    Hebergement = forms.IntegerField(label='Hebergement', required=True)
    Restauration = forms.IntegerField(label='Restauration', required=True)
    code_region = forms.IntegerField(label='Code Région', required=True)
    code_nature = forms.IntegerField(label='Code Nature', required=True)
    statut_public_prive_Public = forms.BooleanField(
        label='Statut Public ?', required=False
    )
