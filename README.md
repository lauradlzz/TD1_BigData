# TD1_BigData — Homework 2.7 et 3.3

Package Python (sans dépendance) qui prend un **JSON Schema** et des **statistiques** en entrée,
construit un objet `Database`, calcule les tailles et la distribution par sharding (2.7),
puis estime le coût des requêtes de filtre et de jointure (3.3).

## Fichiers

| Fichier | Rôle |
|---|---|
| `bigdata_sim/model.py` | Classes `Field` / `Collection` / `Database`, chargement, tailles, sharding |
| `bigdata_sim/costs.py` | Fonctions de coût du cours : volumes réseau / RAM, temps, CO2, prix |
| `bigdata_sim/operators.py` | Opérateurs : filtre et nested loop, avec / sans sharding |
| `schema/db1.json` … `db5.json` | JSON Schemas des 5 dénormalisations |
| `stats.json` | Statistiques de la section 2.2 |
| `cost_params.json` | Constantes du modèle de coût (slides du cours) |
| `main.py` | Affiche les réponses 2.5 et 2.6 pour DB1 à DB5 |
| `queries.py` | Affiche le tableau des coûts de Q1 à Q5 pour DB1 à DB5 (3.1 et 3.2) |
| `test_bigdata_sim.py`, `test_operators.py` | Tests avec valeurs calculées à la main |
| `demo.ipynb` | Démonstration pas à pas (calculs à la main vérifiés par `assert`, graphiques, tests) |

```
python main.py
python queries.py
python -m unittest
```

Utilisation depuis un autre programme :

```python
from bigdata_sim import load_database
db = load_database("schema/db2.json", "stats.json")
db.document_size("Product")           # octets
db.collection_size("Product")         # octets
db.database_size()                    # octets
db.sharding_stats("Product", "brand") # {used_servers, docs_per_server, distinct_per_server}
```

## Conventions

**JSON Schema** : chaque propriété racine est une collection. Types : `integer`, `number`, `string`,
`object`, `array`. Comme JSON Schema ne distingue pas les tailles de chaînes, on utilise
`"format": "date"` (20 B) et `"format": "longstring"` (200 B).

**Statistiques** (`stats.json`) :
- `nb_docs` : nombre d'instances par entité. `Stock` = 10^5 produits × 200 entrepôts = 2·10^7
  (une instance existe même si la quantité est 0).
- `distinct_values` : nombre de valeurs distinctes par clé (pour le sharding).
- `array_lengths` : longueur moyenne d'un tableau quand elle ne se déduit pas des comptes (Categories = 2).

## Formules

Tailles (section 2.5.1) : integer/number 8 B, string 80 B, date 20 B, longstring 200 B, clé 12 B.

- taille(clé) = 12 + taille(valeur)
- taille(objet) = somme des taille(clé) de ses champs
- taille(tableau) = longueur moyenne × taille(élément)
- longueur moyenne d'un tableau d'entité X dans le parent P = nb_docs(X) / nb_docs(P)
  (ex. Stock dans Product : 2·10^7 / 10^5 = 200 ; OrderLine dans Product : 4·10^9 / 10^5 = 40 000)
- taille(collection) = taille(document) × nb_docs ; taille(base) = somme des collections
- 1 GB = 10^9 B

Sharding (section 2.6) :
- serveurs utilisés = min(nb serveurs, nb valeurs distinctes de la clé)
- docs par serveur = nb_docs / serveurs utilisés
- valeurs distinctes par serveur = nb valeurs distinctes / serveurs utilisés

Ex. `St - #IDW` : seulement 200 entrepôts → 200 serveurs utilisés (800 vides), 100 000 docs chacun.

## Choix de modélisation

Dans les documents imbriqués, l'identifiant redondant avec le parent est supprimé
(DB2 `Product.Stock[].IDP`, DB3 `Stock.IDP`, DB4 `OrderLine.IDP`, DB5 `Product.OrderLine[].IDP`).

## Résultats

| Base | Taille |
|---|---|
| DB1 | 1 432,28 GB |
| DB2 | 1 431,88 GB |
| DB3 | 1 455,52 GB |
| DB4 | 6 104,16 GB |
| DB5 | 1 352,28 GB |

| Stratégie | Serveurs utilisés | Docs / serveur | Valeurs distinctes / serveur |
|---|---|---|---|
| St - #IDP | 1 000 | 20 000 | 100 |
| St - #IDW | 200 | 100 000 | 1 |
| OL - #IDC | 1 000 | 4 000 000 | 10 000 |
| OL - #IDP | 1 000 | 4 000 000 | 100 |
| Prod - #IDP | 1 000 | 100 | 100 |
| Prod - #brand | 1 000 | 100 | 5 |

---

# Homework 3.3 — Coût des requêtes

## Utilisation

```python
from bigdata_sim import (load_database, load_cost_params, filter_with_sharding,
                         filter_without_sharding, nested_loop_with_sharding)

db = load_database("schema/db1.json", "stats.json")
params = load_cost_params("cost_params.json")

# Q1 : filtre sur la clé de sharding (St #IDP)
r = filter_with_sharding(db, "Stock", ["IDP", "IDW"], ["quantity", "location"], params)
r["nb_res"], r["size_res"], r["time"], r["co2"], r["price"]

# Q5 : filtre sur Product puis, pour chaque produit, recherche de ses stocks
outer = filter_with_sharding(db, "Product", ["brand"], ["IDP", "name", "price"], params,
                             selectivity={"brand": 50 / 100000})
r = nested_loop_with_sharding(db, outer, "Stock", "IDP", ["IDW", "quantity"], params)
```

Chaque opérateur renvoie un dictionnaire : `S`, `nb_res`, `size_msg`, `size_res`, `nb_docs_read`,
`vol_network`, `vol_ram_max`, `vol_ram_total`, `time` (s), `co2` (kg CO2-eq), `price` (€).

## Paramètres (`cost_params.json`)

| Clé | Valeur | Source |
|---|---|---|
| `bandwidth_network` | 100 MB/s | slide 91 (et slide 110 : RAM = 250 × réseau) |
| `bandwidth_ram` | 25 GB/s | slide 91 |
| `index_size` | 1 MB | slide 96 (index secondaire local) |
| `co2_network` | 0,011 kg CO2-eq / GB | slide 108 |
| `co2_ram` | 0,028 kg CO2-eq / GB | slide 108 |
| `external_fees` | 0,011 € / GB | slide 107 |

## Formules

Variables d'une requête :
- `S` : nombre de serveurs contactés = 1 si le filtre porte sur la clé de sharding, sinon 1 000 (slide 93)
- `size_query` : taille des clés du WHERE (12 B + valeur chacune)
- `size_msg` : taille des clés du SELECT (12 B + valeur chacune)
- `size_doc` : taille d'un document complet (TD1)
- sélectivité d'une clé = 1 / nb valeurs distinctes (ou valeur donnée, ex. Apple = 50 / 10^5) ;
  plusieurs clés (AND) : on multiplie
- `nb_res` = sélectivité × nb lignes (nb lignes = nb éléments du tableau si on renvoie des clés d'un tableau imbriqué)

Filtre (slides 92, 96) :
- `vol_network = S × size_query + nb_res × size_msg`
- `vol_RAM(serveur) = index + documents lus par le serveur × size_doc` (sans index : tous ses documents)

Nested loop (slide 94) : `coût = C1' + |C1'| × C2'` (pour chaque résultat de C1, une recherche dans C2).
"Avec sharding" = C2 est shardée sur la clé de jointure (1 serveur par recherche).

Coûts (slides 106 à 109) :
- temps = `vol_network / bandwidth_network + max(vol_RAM) / bandwidth_ram` (serveurs en parallèle)
- CO2 = `vol_network × co2_network + somme(vol_RAM) × co2_ram` (en GB)
- prix = `vol_network × external_fees` (le prix des serveurs est un coût fixe mensuel, pas par requête)

## Hypothèses

- `size_query` / `size_msg` = clés du WHERE / du SELECT : la slide 96 donne la formule mais pas leur contenu.
- Index de 1 MB lu sur **chaque** serveur contacté (et à chaque recherche d'un nested loop).
- Filtre sur une clé dans un tableau (ex. DB5 Q3, `date` dans `Product.OrderLine[]`) : il ne réduit
  pas le nombre de documents lus (chaque produit a ~110 commandes par jour), seulement les résultats.
- On ne peut pas sharder sur une clé contenue dans un tableau.
- DB3 et DB4 n'ont pas de collection Product : Q2, Q4 et Q5 renvoient des doublons
  (le produit est recopié dans chaque stock / commande), dédoublonnés côté application.
