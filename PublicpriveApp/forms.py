from django import forms

class CandidateForm(forms.Form):
    nb_inscrits = forms.FloatField(label="Nombre d'inscrits")
    nb_presents = forms.FloatField(label="Nombre de présents")
    taux_reussite = forms.FloatField(label="Taux de réussite")
    taux_mentions = forms.FloatField(label="Taux de mentions")
