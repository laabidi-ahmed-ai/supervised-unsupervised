from django.db import models

# Create your models here.
# models.py

class CandidatePerformance(models.Model):
    nb_inscrits = models.IntegerField()
    nb_presents = models.IntegerField()
    taux_reussite = models.FloatField()
    taux_mentions = models.FloatField()
    cluster_hdbscan = models.IntegerField(null=True, blank=True)  # store cluster (-1 = noise)
    
    def __str__(self):
        return f"CandidatePerformance {self.id} - Cluster {self.cluster_hdbscan}"
