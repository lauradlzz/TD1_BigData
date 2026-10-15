"""Tests avec des valeurs calculées à la main. Lancer : python -m unittest"""
import unittest

from bigdata_sim import load_database


def db(i):
    return load_database(f"schema/db{i}.json", "stats.json")


class TestSizes(unittest.TestCase):
    def test_document_sizes_db1(self):
        d = db(1)
        # Product : IDP 20 + name 92 + price (12+20+92+20) + brand 92 + description 212
        #           + image_url 92 + Categories (12 + 2*92) + Supplier (12+20+92+92+92+20) = 1176
        self.assertEqual(d.document_size("Product"), 1176)
        self.assertEqual(d.document_size("Stock"), 152)       # 20+20+20+92
        self.assertEqual(d.document_size("Warehouse"), 132)   # 20+92+20
        self.assertEqual(d.document_size("OrderLine"), 356)   # 20+20+32+20+32+212+20
        self.assertEqual(d.document_size("Client"), 512)      # 20+5*92+32

    def test_nested_arrays(self):
        # DB2 : Product + Stock[200 x (IDW 20 + quantity 20 + location 92)] + 12
        self.assertEqual(db(2).document_size("Product"), 1176 + 12 + 200 * 132)
        # DB5 : Product + OrderLine[40 000 x 336] + 12
        self.assertEqual(db(5).document_size("Product"), 1176 + 12 + 40_000 * 336)

    def test_nested_objects(self):
        # DB3 : Stock sans IDP (132) + clé Product (12 + 1176)
        self.assertEqual(db(3).document_size("Stock"), 132 + 12 + 1176)
        # DB4 : OrderLine sans IDP (336) + clé Product (12 + 1176)
        self.assertEqual(db(4).document_size("OrderLine"), 336 + 12 + 1176)

    def test_database_size_db1(self):
        expected = 1176 * 10**5 + 152 * 2 * 10**7 + 132 * 200 + 356 * 4 * 10**9 + 512 * 10**7
        self.assertEqual(db(1).database_size(), expected)


class TestSharding(unittest.TestCase):
    def test_more_values_than_servers(self):
        s = db(1).sharding_stats("Stock", "IDP")
        self.assertEqual(s["used_servers"], 1000)
        self.assertEqual(s["docs_per_server"], 20_000)
        self.assertEqual(s["distinct_per_server"], 100)

    def test_fewer_values_than_servers(self):
        # 200 entrepôts pour 1000 serveurs : 800 serveurs vides
        s = db(1).sharding_stats("Stock", "IDW")
        self.assertEqual(s["used_servers"], 200)
        self.assertEqual(s["docs_per_server"], 100_000)
        self.assertEqual(s["distinct_per_server"], 1)

    def test_nested_key(self):
        # DB4 : brand est dans OrderLine.Product
        s = db(4).sharding_stats("OrderLine", "brand")
        self.assertEqual(s["docs_per_server"], 4_000_000)

    def test_unknown_key(self):
        with self.assertRaises(KeyError):
            db(1).sharding_stats("Warehouse", "IDP")


if __name__ == "__main__":
    unittest.main()
