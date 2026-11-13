"""
Homework 3.3 : fonctions de coût du cours (slides 91 à 110).
Toutes les tailles sont en octets, les paramètres viennent de cost_params.json :
  - bandwidth_network : 100 MB/s   (slide 91, = bandwidth_ram / 250 slide 110)
  - bandwidth_ram     : 25 GB/s    (slide 91)
  - index_size        : 1 MB       (index secondaire local, slide 96)
  - co2_network       : 0,011 kg CO2-eq / GB  (slide 108)
  - co2_ram           : 0,028 kg CO2-eq / GB  (slide 108)
  - external_fees     : 0,011 € / GB          (slide 107)
"""
import json

from .model import GB


def load_cost_params(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- Volumes ----------

def vol_network(S, size_query, nb_res, size_msg):
    """Slide 96 : la requête est envoyée à S serveurs, puis les résultats reviennent."""
    return S * size_query + nb_res * size_msg


def vol_ram(index_size, nb_docs_read, size_doc):
    """Slide 96 : volume lu par UN serveur = index + documents qui passent le filtre."""
    return index_size + nb_docs_read * size_doc


# ---------- Coûts ----------

def time_cost(vol_net, vol_ram_max, params):
    """Slide 106 : temps réseau + temps RAM.
    Les serveurs travaillent en parallèle, donc on prend le serveur le plus chargé."""
    return vol_net / params["bandwidth_network"] + vol_ram_max / params["bandwidth_ram"]


def carbon_cost(vol_net, vol_ram_total, params):
    """Slide 109 : kg CO2-eq. Pour la RAM on fait la SOMME sur tous les serveurs."""
    return vol_net / GB * params["co2_network"] + vol_ram_total / GB * params["co2_ram"]


def price_cost(vol_net, params):
    """Slide 107 : frais de communication externe (application <-> cluster), en €.
    Le prix des serveurs est un coût fixe mensuel de la base, pas un coût par requête."""
    return vol_net / GB * params["external_fees"]
