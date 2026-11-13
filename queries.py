"""Répond aux sections 3.1 et 3.2 : coûts des requêtes Q1 à Q5 sur DB1 à DB5."""
from bigdata_sim import (load_database, load_cost_params,
                         filter_with_sharding, filter_without_sharding,
                         nested_loop_with_sharding, nested_loop_without_sharding)

params = load_cost_params("cost_params.json")
db = {i: load_database(f"schema/db{i}.json", "stats.json") for i in range(1, 6)}

# Q2 / Q5 : le sujet donne 50 produits Apple (au lieu de 10^5 / 5 000 = 20 en moyenne)
APPLE = {"brand": 50 / 100000}


def show(query, db_name, sharding, algo, r):
    print(f"{query:<3} {db_name:<4} {sharding:<26} {algo:<34} {r['nb_res']:>13,.0f}"
          f" {r['vol_network'] / 1e6:>12,.3f} {r['time']:>12,.4f} {r['co2']:>12,.6f} {r['price']:>12,.7f}")


def header(title):
    print(f"\n=== {title} ===")
    print(f"{'Q':<3} {'DB':<4} {'Sharding':<26} {'Algo':<34} {'nb résultats':>13}"
          f" {'réseau (MB)':>12} {'temps (s)':>12} {'CO2 (kg)':>12} {'prix (€)':>12}")


if __name__ == "__main__":
    # ---------- Q1 : stock d'un produit dans un entrepôt ----------
    header("Q1 : SELECT quantity, location FROM Stock WHERE IDP = $IDP AND IDW = $IDW")
    out = ["quantity", "location"]
    for i in (1, 4, 5):
        show("Q1", f"DB{i}", "St #IDP", "Shard + Index",
             filter_with_sharding(db[i], "Stock", ["IDP", "IDW"], out, params))
        show("Q1", f"DB{i}", "St #_id (autre clé)", "Broadcast + Index",
             filter_without_sharding(db[i], "Stock", ["IDP", "IDW"], out, params))
    # DB2 : le stock est dans un tableau du produit
    show("Q1", "DB2", "Prod #IDP", "Shard + Index",
         filter_with_sharding(db[2], "Product", ["IDP", "IDW"], out, params))
    show("Q1", "DB2", "Prod #brand", "Broadcast + Index",
         filter_without_sharding(db[2], "Product", ["IDP", "IDW"], out, params))
    # DB3 : IDP est dans le Product imbriqué
    show("Q1", "DB3", "St #Product.IDP", "Shard + Index",
         filter_with_sharding(db[3], "Stock", ["IDP", "IDW"], out, params))
    show("Q1", "DB3", "St #_id (autre clé)", "Broadcast + Index",
         filter_without_sharding(db[3], "Stock", ["IDP", "IDW"], out, params))

    # ---------- Q2 : produits d'une marque ----------
    header("Q2 : SELECT name, price FROM Product WHERE brand = 'Apple'")
    out = ["name", "price"]
    for i in (1, 2, 5):
        show("Q2", f"DB{i}", "Prod #brand", "Shard + Index",
             filter_with_sharding(db[i], "Product", ["brand"], out, params, APPLE))
        show("Q2", f"DB{i}", "Prod #IDP", "Broadcast + Index",
             filter_without_sharding(db[i], "Product", ["brand"], out, params, APPLE))
    # DB3 / DB4 : pas de collection Product, on filtre le produit imbriqué (avec doublons)
    show("Q2", "DB3", "St #Product.brand", "Shard + Index (doublons)",
         filter_with_sharding(db[3], "Stock", ["brand"], out, params, APPLE))
    show("Q2", "DB3", "St #Product.IDP", "Broadcast + Index (doublons)",
         filter_without_sharding(db[3], "Stock", ["brand"], out, params, APPLE))
    show("Q2", "DB4", "OL #Product.brand", "Shard + Index (doublons)",
         filter_with_sharding(db[4], "OrderLine", ["brand"], out, params, APPLE))
    show("Q2", "DB4", "OL #IDC", "Broadcast + Index (doublons)",
         filter_without_sharding(db[4], "OrderLine", ["brand"], out, params, APPLE))

    # ---------- Q3 : lignes de commande d'une date ----------
    header("Q3 : SELECT IDP, quantity FROM OrderLine WHERE date = $date")
    out = ["IDP", "quantity"]
    for i in (1, 2, 3, 4):
        show("Q3", f"DB{i}", "OL #date", "Shard + Index",
             filter_with_sharding(db[i], "OrderLine", ["date"], out, params))
        show("Q3", f"DB{i}", "OL #IDC", "Broadcast + Index",
             filter_without_sharding(db[i], "OrderLine", ["date"], out, params))
    # DB5 : les commandes sont dans un tableau du produit, on ne peut pas sharder sur date
    show("Q3", "DB5", "Prod #IDP", "Broadcast + Index (tableau)",
         filter_without_sharding(db[5], "Product", ["date"], out, params))

    # ---------- Q4 : stock d'un entrepôt (jointure Stock -> Product) ----------
    header("Q4 : SELECT P.name, S.quantity FROM Stock S JOIN Product P ON S.IDP = P.IDP WHERE S.IDW = $IDW")
    for i in (1, 5):
        outer = filter_with_sharding(db[i], "Stock", ["IDW"], ["IDP", "quantity"], params)
        show("Q4", f"DB{i}", "St #IDW, Prod #IDP", "Nested loop avec sharding",
             nested_loop_with_sharding(db[i], outer, "Product", "IDP", ["name"], params))
        outer = filter_without_sharding(db[i], "Stock", ["IDW"], ["IDP", "quantity"], params)
        show("Q4", f"DB{i}", "St #IDP, Prod #brand", "Nested loop sans sharding",
             nested_loop_without_sharding(db[i], outer, "Product", "IDP", ["name"], params))
    # DB2 : le stock est dans le produit, la jointure est déjà faite (filtre sur le tableau)
    show("Q4", "DB2", "Prod #IDP", "Broadcast + Index (tableau)",
         filter_without_sharding(db[2], "Product", ["IDW"], ["name", "quantity"], params))
    # DB3 : le produit est dans le stock, la jointure est déjà faite
    show("Q4", "DB3", "St #IDW", "Shard + Index",
         filter_with_sharding(db[3], "Stock", ["IDW"], ["name", "quantity"], params))
    show("Q4", "DB3", "St #Product.IDP", "Broadcast + Index",
         filter_without_sharding(db[3], "Stock", ["IDW"], ["name", "quantity"], params))
    # DB4 : pas de collection Product, il faut aller chercher le nom dans OrderLine (doublons)
    outer = filter_with_sharding(db[4], "Stock", ["IDW"], ["IDP", "quantity"], params)
    show("Q4", "DB4", "St #IDW, OL #Product.IDP", "Nested loop avec sharding (doublons)",
         nested_loop_with_sharding(db[4], outer, "OrderLine", "IDP", ["name"], params))

    # ---------- Q5 : répartition des produits Apple dans les entrepôts ----------
    header("Q5 : SELECT P.name, P.price, S.IDW, S.quantity FROM Product P JOIN Stock S ... WHERE brand = 'Apple'")
    for i in (1, 5):
        outer = filter_with_sharding(db[i], "Product", ["brand"], ["IDP", "name", "price"], params, APPLE)
        show("Q5", f"DB{i}", "Prod #brand, St #IDP", "Nested loop avec sharding",
             nested_loop_with_sharding(db[i], outer, "Stock", "IDP", ["IDW", "quantity"], params))
        outer = filter_without_sharding(db[i], "Product", ["brand"], ["IDP", "name", "price"], params, APPLE)
        show("Q5", f"DB{i}", "Prod #IDP, St #IDW", "Nested loop sans sharding",
             nested_loop_without_sharding(db[i], outer, "Stock", "IDP", ["IDW", "quantity"], params))
    # DB2 : les stocks sont dans le produit
    out = ["name", "price", "IDW", "quantity"]
    show("Q5", "DB2", "Prod #brand", "Shard + Index",
         filter_with_sharding(db[2], "Product", ["brand"], out, params, APPLE))
    show("Q5", "DB2", "Prod #IDP", "Broadcast + Index",
         filter_without_sharding(db[2], "Product", ["brand"], out, params, APPLE))
    # DB3 : le produit est dans chaque stock
    show("Q5", "DB3", "St #Product.brand", "Shard + Index",
         filter_with_sharding(db[3], "Stock", ["brand"], out, params, APPLE))
    show("Q5", "DB3", "St #IDW", "Broadcast + Index",
         filter_without_sharding(db[3], "Stock", ["brand"], out, params, APPLE))
    # DB4 : pas de collection Product, on part des OrderLine Apple (doublons)
    outer = filter_with_sharding(db[4], "OrderLine", ["brand"], ["IDP", "name", "price"], params, APPLE)
    show("Q5", "DB4", "OL #Product.brand, St #IDP", "Nested loop avec sharding (doublons)",
         nested_loop_with_sharding(db[4], outer, "Stock", "IDP", ["IDW", "quantity"], params))
