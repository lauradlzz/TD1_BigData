# TD1_BigData — Homework 2.7

Package Python (sans dépendance) qui prend un **JSON Schema** et des **statistiques** en entrée,
construit un objet `Database`, puis calcule les tailles et la distribution par sharding.

## Fichiers

| Fichier | Rôle |
|---|---|
| `bigdata_sim/model.py` | Classes `Field` / `Collection` / `Database`, chargement, tailles, sharding |
| `schema/db1.json` … `db5.json` | JSON Schemas des 5 dénormalisations |
| `stats.json` | Statistiques de la section 2.2 |
| `main.py` | Affiche les réponses 2.5 et 2.6 pour DB1 à DB5 |
| `test_bigdata_sim.py` | Tests avec valeurs calculées à la main |
| `demo.ipynb` | Démonstration pas à pas (calculs à la main vérifiés par `assert`, graphique, tests) |

```
python main.py
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
