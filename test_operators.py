"""Tests des opérateurs (TD 3.3) avec des valeurs calculées à la main. Lancer : python -m unittest"""
import unittest

from bigdata_sim import (load_database, load_cost_params,
                         filter_with_sharding, filter_without_sharding,
                         nested_loop_with_sharding)
from bigdata_sim.operators import keys_size

params = load_cost_params("cost_params.json")


def db(i):
    return load_database(f"schema/db{i}.json", "stats.json")


class TestSizes(unittest.TestCase):
    def test_query_and_msg_sizes(self):
        d = db(1)
        self.assertEqual(keys_size(d, "Stock", ["IDP", "IDW"]), 40)             # 20 + 20
        self.assertEqual(keys_size(d, "Stock", ["quantity", "location"]), 112)  # 20 + 92
        self.assertEqual(keys_size(d, "Product", ["name", "price"]), 236)       # 92 + 144

    def test_key_in_nested_object_and_array(self):
        # DB3 : name est dans Stock.Product (et pas celui de Supplier)
        self.assertEqual(keys_size(db(3), "Stock", ["name"]), 92)
        # DB2 : quantity est dans le tableau Product.Stock
        self.assertEqual(keys_size(db(2), "Product", ["quantity"]), 20)

    def test_unknown_key(self):
        with self.assertRaises(KeyError):
            filter_with_sharding(db(1), "Warehouse", ["brand"], ["IDW"], params)


class TestFilter(unittest.TestCase):
    def test_q1_with_sharding(self):
        r = filter_with_sharding(db(1), "Stock", ["IDP", "IDW"], ["quantity", "location"], params)
        self.assertAlmostEqual(r["nb_res"], 1)
        self.assertAlmostEqual(r["vol_network"], 152)            # 1 x 40 + 1 x 112
        self.assertAlmostEqual(r["vol_ram_max"], 1_000_152)      # 1 MB d'index + 1 x 152
        self.assertAlmostEqual(r["time"], 152 / 1e8 + 1_000_152 / 25e9)
        self.assertAlmostEqual(r["co2"], 152e-9 * 0.011 + 1_000_152e-9 * 0.028)

    def test_q1_without_sharding(self):
        r = filter_without_sharding(db(1), "Stock", ["IDP", "IDW"], ["quantity", "location"], params)
        self.assertAlmostEqual(r["vol_network"], 40_112)         # 1000 x 40 + 1 x 112
        self.assertAlmostEqual(r["vol_ram_total"], 1_000_000_152)  # 1000 index + 1 doc
        self.assertAlmostEqual(r["co2"], 40_112e-9 * 0.011 + 1_000_000_152e-9 * 0.028)

    def test_q2_apple(self):
        r = filter_with_sharding(db(1), "Product", ["brand"], ["name", "price"], params,
                                 selectivity={"brand": 50 / 100000})
        self.assertAlmostEqual(r["nb_res"], 50)
        self.assertAlmostEqual(r["vol_network"], 11_892)         # 92 + 50 x 236
        self.assertAlmostEqual(r["vol_ram_max"], 1_058_800)      # 1 MB + 50 x 1176

    def test_default_selectivity(self):
        # Marque moyenne : 10^5 / 5000 = 20 produits
        r = filter_with_sharding(db(1), "Product", ["brand"], ["name"], params)
        self.assertAlmostEqual(r["nb_res"], 20)

    def test_filter_in_array_db2(self):
        # DB2 Q1 : on lit 1 Product entier (27 588 B) pour renvoyer 1 élément de son tableau
        r = filter_with_sharding(db(2), "Product", ["IDP", "IDW"], ["quantity", "location"], params)
        self.assertAlmostEqual(r["nb_res"], 1)
        self.assertAlmostEqual(r["nb_docs_read"], 1)
        self.assertAlmostEqual(r["vol_ram_max"], 1_000_000 + 27_588)

    def test_filter_in_array_db5(self):
        # DB5 Q3 : 4.10^9 / 365 lignes renvoyées, mais les 10^5 produits sont lus
        r = filter_without_sharding(db(5), "Product", ["date"], ["IDP", "quantity"], params)
        self.assertAlmostEqual(r["nb_res"], 4e9 / 365)
        self.assertAlmostEqual(r["nb_docs_read"], 100_000)

    def test_full_scan(self):
        # Sans index : chaque serveur lit ses 20 000 stocks
        r = filter_without_sharding(db(1), "Stock", ["IDW"], ["quantity"], params, index=False)
        self.assertAlmostEqual(r["vol_ram_max"], 20_000 * 152)
        self.assertAlmostEqual(r["vol_ram_total"], 1000 * 20_000 * 152)


class TestNestedLoop(unittest.TestCase):
    def test_q5_db1_with_sharding(self):
        d = db(1)
        outer = filter_with_sharding(d, "Product", ["brand"], ["IDP", "name", "price"], params,
                                     selectivity={"brand": 50 / 100000})
        r = nested_loop_with_sharding(d, outer, "Stock", "IDP", ["IDW", "quantity"], params)
        # outer : 92 + 50 x 256 = 12 892 B ; inner (1 recherche) : 20 + 200 x 40 = 8 020 B
        self.assertAlmostEqual(r["nb_res"], 50 * 200)
        self.assertAlmostEqual(r["vol_network"], 12_892 + 50 * 8_020)
        # outer : 1 MB + 50 x 1176 ; inner : 1 MB + 200 x 152
        self.assertAlmostEqual(r["vol_ram_max"], 1_058_800 + 50 * 1_030_400)


if __name__ == "__main__":
    unittest.main()
