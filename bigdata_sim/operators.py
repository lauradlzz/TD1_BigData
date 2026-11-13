"""
Homework 3.3 : opérateurs de filtre et de jointure (nested loop), avec ou sans sharding.
Chaque opérateur renvoie un dictionnaire avec le nombre de résultats, leur taille,
les volumes (réseau / RAM) et les trois coûts (temps, CO2, prix).
"""
from .costs import vol_network, vol_ram, time_cost, carbon_cost, price_cost


def find_key(fields, name, array=None):
    """Cherche une clé partout dans le document, y compris dans les tableaux.
    Renvoie (champ, tableau qui contient la clé ou None).
    Si une clé existe plusieurs fois (ex. name dans Product et dans Supplier),
    c'est la première rencontrée qui est renvoyée (celle du niveau le plus haut)."""
    for f in fields:
        if f.name == name:
            return f, array
    for f in fields:
        if f.kind == "object":
            found = find_key(f.children, name, array)
        elif f.kind == "array" and f.item.kind == "object":
            found = find_key(f.item.children, name, f)
        else:
            continue
        if found[0] is not None:
            return found
    return None, None


def get_key(db, coll_name, name):
    field, array = find_key(db.collections[coll_name].fields, name)
    if field is None:
        raise KeyError(f"La clé '{name}' n'existe pas dans {coll_name} ({db.name})")
    return field, array


def keys_size(db, coll_name, keys):
    """Taille d'un ensemble de clés (12B + valeur pour chacune).
    Sert pour size_query (clés du WHERE) et size_msg (clés du SELECT)."""
    return sum(db.field_size(get_key(db, coll_name, k)[0], coll_name) for k in keys)


# ==========================================
# FILTRE
# ==========================================

def filter_op(db, coll_name, filter_keys, output_keys, S, params, selectivity=None, index=True):
    """Filtre sur une collection, envoyé à S serveurs.
    - filter_keys : clés du WHERE (égalités)
    - output_keys : clés du SELECT
    - selectivity : dict optionnel {clé: sélectivité}, sinon 1 / nb valeurs distinctes
    - index       : True = index local sur la clé filtrée, False = parcours complet"""
    coll = db.collections[coll_name]
    selectivity = selectivity or {}

    # Sélectivité de chaque clé filtrée (valeurs supposées uniformes)
    sel_total = 1      # toutes les clés
    sel_root = 1       # seulement les clés hors tableau
    for k in filter_keys:
        _, array = get_key(db, coll_name, k)
        s = selectivity.get(k, 1 / db.stats["distinct_values"][k])
        sel_total *= s
        if array is None:
            sel_root *= s

    # Nombre de "lignes" candidates : si on renvoie des clés d'un tableau imbriqué,
    # chaque élément du tableau est une ligne (ex. DB5 : 10^5 Product x 40 000 OrderLine)
    nb_lines = coll.nb_docs
    for k in output_keys:
        _, array = get_key(db, coll_name, k)
        if array is not None:
            nb_lines = coll.nb_docs * db.avg_array_length(array.name, coll_name)
            break

    nb_res = sel_total * nb_lines
    # Documents lus : seuls les filtres hors tableau réduisent le nombre de documents.
    # Un filtre dans un tableau garde presque tous les documents (ex. DB5 Q3 : chaque
    # produit a ~110 commandes par jour), mais on ne lit jamais plus de docs que de résultats.
    nb_docs_read = min(coll.nb_docs * sel_root, nb_res)

    size_query = keys_size(db, coll_name, filter_keys)
    size_msg = keys_size(db, coll_name, output_keys)
    size_doc = db.document_size(coll_name)

    if index:
        # Chaque serveur lit son index + sa part des documents trouvés
        ram_max = vol_ram(params["index_size"], nb_docs_read / S, size_doc)
        ram_total = S * params["index_size"] + nb_docs_read * size_doc
    else:
        # Parcours complet : chaque serveur lit tous ses documents
        local_docs = coll.nb_docs / db.stats["servers"]
        ram_max = local_docs * size_doc
        ram_total = S * ram_max

    net = vol_network(S, size_query, nb_res, size_msg)
    return {
        "S": S,
        "nb_res": nb_res,
        "size_msg": size_msg,
        "size_res": nb_res * size_msg,
        "nb_docs_read": nb_docs_read,
        "vol_network": net,
        "vol_ram_max": ram_max,
        "vol_ram_total": ram_total,
        "time": time_cost(net, ram_max, params),
        "co2": carbon_cost(net, ram_total, params),
        "price": price_cost(net, params),
    }


def filter_with_sharding(db, coll_name, filter_keys, output_keys, params, selectivity=None, index=True):
    """Le filtre porte sur la clé de sharding : un seul serveur est contacté (S = 1)."""
    return filter_op(db, coll_name, filter_keys, output_keys, 1, params, selectivity, index)


def filter_without_sharding(db, coll_name, filter_keys, output_keys, params, selectivity=None, index=True):
    """Le filtre ne porte pas sur la clé de sharding : tous les serveurs sont contactés."""
    S = db.stats["servers"]
    return filter_op(db, coll_name, filter_keys, output_keys, S, params, selectivity, index)


# ==========================================
# JOINTURE : NESTED LOOP (slide 94)
# ==========================================

def nested_loop(outer, inner, params):
    """Combine un filtre sur C1 (outer) avec la recherche dans C2 (inner) faite
    pour CHAQUE résultat de C1 : coût = C1' + |C1'| x C2'  (slides 94 et 96).
    Les recherches dans C2 sont faites les unes après les autres, donc les volumes s'additionnent."""
    n = outer["nb_res"]
    net = outer["vol_network"] + n * inner["vol_network"]
    ram_max = outer["vol_ram_max"] + n * inner["vol_ram_max"]
    ram_total = outer["vol_ram_total"] + n * inner["vol_ram_total"]
    return {
        "S": outer["S"] + n * inner["S"],   # nombre total de serveurs contactés
        "nb_res": n * inner["nb_res"],
        "size_msg": outer["size_msg"] + inner["size_msg"],
        "size_res": n * inner["nb_res"] * (outer["size_msg"] + inner["size_msg"]),
        "nb_docs_read": outer["nb_docs_read"] + n * inner["nb_docs_read"],
        "vol_network": net,
        "vol_ram_max": ram_max,
        "vol_ram_total": ram_total,
        "time": time_cost(net, ram_max, params),
        "co2": carbon_cost(net, ram_total, params),
        "price": price_cost(net, params),
    }


def nested_loop_with_sharding(db, outer, coll2, join_key, output_keys2, params):
    """C2 est shardée sur la clé de jointure : chaque recherche dans C2 touche 1 serveur.
    outer = résultat d'un filtre sur C1 (ses clés de sortie doivent contenir join_key)."""
    inner = filter_with_sharding(db, coll2, [join_key], output_keys2, params)
    return nested_loop(outer, inner, params)


def nested_loop_without_sharding(db, outer, coll2, join_key, output_keys2, params):
    """C2 n'est pas shardée sur la clé de jointure : chaque recherche touche tous les serveurs."""
    inner = filter_without_sharding(db, coll2, [join_key], output_keys2, params)
    return nested_loop(outer, inner, params)
