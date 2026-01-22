from django import forms

class ClusteringForm(forms.Form):
    # Colonnes que l'utilisateur peut remplir
    SEGPA = forms.FloatField(label="SEGPA", required=True)
    Hebergement = forms.FloatField(label="Hebergement", required=True)
    Restauration = forms.FloatField(label="Restauration", required=True)
    ulis = forms.FloatField(label="ULIS", required=True)
    ecole_elementaire = forms.FloatField(label="Ecole Élémentaire", required=False, initial=0.0)
    ecole_maternelle = forms.FloatField(label="Ecole Maternelle", required=False, initial=0.0)
    code_nature = forms.FloatField(label="Code Nature", required=True)
    code_region = forms.FloatField(label="Code Région", required=True)
    longitude = forms.FloatField(label="Longitude", required=False, initial=0.0)
    latitude = forms.FloatField(label="Latitude", required=False, initial=0.0)
    code_type_contrat_prive = forms.FloatField(label="Type Contrat Privé", required=False, initial=0.0)
    id_academie = forms.FloatField(label="ID Académie", required=False, initial=0.0)
    nb_inscrits = forms.FloatField(label="Nombre d'inscrits", required=False, initial=0.0)
    nb_admis_1er_groupe = forms.FloatField(label="Admis 1er groupe", required=False, initial=0.0)
    nb_refuses_1er_groupe = forms.FloatField(label="Refusés 1er groupe", required=False, initial=0.0)
    nb_admis_totaux = forms.FloatField(label="Admis totaux", required=False, initial=0.0)
    nb_refuses_totaux = forms.FloatField(label="Refusés totaux", required=False, initial=0.0)
    taux_reussite = forms.FloatField(label="Taux de réussite", required=False, initial=0.0)
    taux_1er_groupe = forms.FloatField(label="Taux 1er groupe", required=False, initial=0.0)
    taux_echec = forms.FloatField(label="Taux d'échec", required=False, initial=0.0)
    
    # Types d'établissement
    type_etablissement_Collège = forms.FloatField(label="Collège", required=False, initial=0.0)
    type_etablissement_EREA = forms.FloatField(label="EREA", required=False, initial=0.0)
    type_etablissement_Ecole = forms.FloatField(label="École", required=False, initial=0.0)
    type_etablissement_Inconnu = forms.FloatField(label="Inconnu", required=False, initial=0.0)
    type_etablissement_Information_et_orientation = forms.FloatField(
        label="Information et orientation", required=False, initial=0.0
    )
    type_etablissement_Lycée = forms.FloatField(label="Lycée", required=False, initial=0.0)
    type_etablissement_Médico_social = forms.FloatField(label="Médico-social", required=False, initial=0.0)
    type_etablissement_Service_Administratif = forms.FloatField(
        label="Service Administratif", required=False, initial=0.0
    )
    
    # Statut public/privé
    statut_public_prive_Privé = forms.FloatField(label="Statut Privé", required=False, initial=0.0)
    
    # Optionnel : score et date_sk_x si tu veux laisser remplir
    score = forms.FloatField(label="Score", required=False, initial=0.0)
    date_sk_x = forms.FloatField(label="Date (sk)", required=False, initial=0.0)
