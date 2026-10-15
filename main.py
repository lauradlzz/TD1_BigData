"""Répond aux questions 2.5 et 2.6 pour DB1 à DB5."""
from bigdata_sim import load_database
from bigdata_sim.model import GB

SHARDING_STRATEGIES = [
    ("Stock", "IDP"),
    ("Stock", "IDW"),
    ("OrderLine", "IDC"),
    ("OrderLine", "IDP"),
    ("Product", "IDP"),
    ("Product", "brand"),
]

if __name__ == "__main__":
    # 2.5 : tailles
    for i in range(1, 6):
        db = load_database(f"schema/db{i}.json", "stats.json")
        print(f"=== {db.name} ===")
        for name, coll in db.collections.items():
            print(f"  {name:<10} doc = {db.document_size(name):>14,.0f} B"
                  f" | nb docs = {coll.nb_docs:>13,}"
                  f" | collection = {db.collection_size(name) / GB:>12,.4f} GB")
        print(f"  TOTAL : {db.database_size() / GB:,.4f} GB\n")

    # 2.6 : sharding (collections de DB1)
    db1 = load_database("schema/db1.json", "stats.json")
    print("=== Sharding (DB1, 1000 serveurs) ===")
    for coll, key in SHARDING_STRATEGIES:
        s = db1.sharding_stats(coll, key)
        print(f"  {coll + ' - #' + key:<18} serveurs utilisés = {s['used_servers']:>5}"
              f" | docs/serveur = {s['docs_per_server']:>12,.0f}"
              f" | valeurs distinctes/serveur = {s['distinct_per_server']:>8,.0f}")
