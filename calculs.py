import json

class DataModelSimulator:
    def __init__(self, schema_dict, stats_dict):
        """
        Génère la structure (objet) à manipuler pour les futurs programmes (Homework 2.7)
        Prend en entrée le schéma JSON et les statistiques.
        """
        self.schema = schema_dict
        self.stats = stats_dict
        
        # Définition des tailles en octets (Bytes) selon la section 2.5
        self.type_sizes = {
            "Integer": 8,
            "Number": 8,
            "String": 80,
            "Date": 20,
            "LongString": 200
        }
        self.key_overhead = 12 # "Key+Value pairs/Arrays: 12B + values"
        
        # Constante GigaByte
        self.GB = 1024 ** 3

        # Calcul du nombre total de documents par collection d'après les statistiques de la section 2.2
        # - Stock : Même si la quantité est 0, il y a une instance (10^5 produits * 200 entrepôts)
        self.collection_counts = {
            "Product": self.stats["products"],
            "Stock": self.stats["products"] * self.stats["warehouses"], 
            "Warehouse": self.stats["warehouses"],
            "OrderLine": self.stats["order_lines"],
            "Client": self.stats["clients"]
        }

        # Mapping des clés de sharding à leur nombre de valeurs distinctes
        self.distinct_values = {
            "IDP": self.stats["products"],
            "IDW": self.stats["warehouses"],
            "IDC": self.stats["clients"],
            "brand": self.stats["distinct_brands"]
        }

    def _calculate_node_size(self, node, key_name=None, is_root=False):
        """Fonction récursive pour calculer la taille d'un document ou sous-document JSON."""
        size = 0
        
        # Ajout du coût de la clé (12B) si ce n'est pas la racine du document
        if not is_root and key_name is not None:
            size += self.key_overhead
            
        if isinstance(node, str):
            # C'est un type primitif (ex: "String", "Integer")
            size += self.type_sizes.get(node, 0)
            
        elif isinstance(node, dict):
            # C'est un objet JSON
            for k, v in node.items():
                size += self._calculate_node_size(v, key_name=k, is_root=False)
                
        elif isinstance(node, list):
            # C'est un tableau (Array)
            multiplier = 1
            # Gestion des multiplicateurs basés sur les statistiques pour les tableaux imbriqués
            if key_name == "Categories":
                multiplier = self.stats.get("categories_per_product_avg", 2)
                
            if len(node) > 0:
                element_size = self._calculate_node_size(node[0], key_name=None, is_root=True)
                size += multiplier * element_size
                
        return size

    # --- FONCTIONS DE CALCUL DES TAILLES (Section 2.5 & 2.7) ---

    def compute_document_size(self, collection_name):
        """Calcule la taille moyenne d'un document pour une collection donnée en Octets (Bytes)."""
        if collection_name not in self.schema:
            return 0
        return self._calculate_node_size(self.schema[collection_name], is_root=True)

    def compute_collection_size(self, collection_name):
        """Calcule la taille totale d'une collection en Octets (Bytes)."""
        doc_size = self.compute_document_size(collection_name)
        count = self.collection_counts.get(collection_name, 0)
        return doc_size * count
        
    def compute_database_size(self):
        """Calcule la taille totale de la base de données en Octets (Bytes)."""
        total_size = 0
        for coll in self.schema.keys():
            total_size += self.compute_collection_size(coll)
        return total_size

    def print_sizes_report(self):
        """Affiche un rapport des tailles en Gigaoctets (GB)."""
        print("=== RAPPORT DE TAILLES (DB1) ===")
        for coll in self.schema.keys():
            size_b = self.compute_collection_size(coll)
            doc_size = self.compute_document_size(coll)
            count = self.collection_counts.get(coll, 0)
            print(f"- Collection '{coll}' : {size_b / self.GB:.4f} GB")
            print(f"    -> Doc Size: {doc_size} B | Nb Docs: {count:,}")
        
        print(f"\n=> TAILLE TOTALE DE LA BASE : {self.compute_database_size() / self.GB:.4f} GB\n")

    # --- FONCTIONS DE SHARDING (Section 2.6 & 2.7) ---

    def compute_sharding_stats(self, collection_name, sharding_key):
        """Calcule les statistiques de distribution pour une stratégie de sharding donnée."""
        servers = self.stats["servers"]
        total_docs = self.collection_counts.get(collection_name, 0)
        
        # Nettoyage de la clé (ex: "#IDP" -> "IDP")
        clean_key = sharding_key.replace("#", "")
        distinct_vals = self.distinct_values.get(clean_key, 0)
        
        docs_per_server = total_docs / servers
        distinct_per_server = distinct_vals / servers
        
        return {
            "collection": collection_name,
            "sharding_key": sharding_key,
            "docs_per_server": docs_per_server,
            "distinct_values_per_server": distinct_per_server
        }

    def print_sharding_report(self, strategies):
        """Affiche le rapport pour la question 2.6."""
        print("=== RAPPORT DE STRATÉGIES DE SHARDING (1000 serveurs) ===")
        for coll, key in strategies:
            stats = self.compute_sharding_stats(coll, key)
            print(f"Stratégie : {coll} - {key}")
            print(f"  -> Docs par serveur (moyenne) : {stats['docs_per_server']:,.0f}")
            print(f"  -> Valeurs distinctes par serveur : {stats['distinct_values_per_server']:,.1f}\n")


# ==========================================
# DONNÉES D'ENTRÉE (JSON Schema & Stats)
# ==========================================

stats_json_str = """
{
  "clients": 10000000,
  "products": 100000,
  "order_lines": 4000000000,
  "warehouses": 200,
  "categories_per_product_avg": 2,
  "distinct_brands": 5000,
  "servers": 1000
}
"""

schema_db1_json_str = """
{
  "Product": {
    "IDP": "Integer",
    "name": "String",
    "price": {
      "amount": "Integer",
      "currency": "String",
      "vat": "Integer"
    },
    "brand": "String",
    "description": "LongString",
    "image_url": "String",
    "Categories": [
      {
        "title": "String"
      }
    ],
    "Supplier": {
      "IDS": "Integer",
      "name": "String",
      "SIRET": "String",
      "headOffice": "String",
      "Revenue": "Integer"
    }
  },
  "Stock": {
    "IDP": "Integer",
    "IDW": "Integer",
    "quantity": "Integer",
    "location": "String"
  },
  "Warehouse": {
    "IDW": "Integer",
    "address": "String",
    "capacity": "Integer"
  },
  "OrderLine": {
    "IDC": "Integer",
    "IDP": "Integer",
    "date": "Date",
    "quantity": "Integer",
    "deliveryDate": "Date",
    "comment": "LongString",
    "grade": "Integer"
  },
  "Client": {
    "IDC": "Integer",
    "ln": "String",
    "fn": "String",
    "address": "String",
    "nationality": "String",
    "birthDate": "Date",
    "email": "String"
  }
}
"""

# ==========================================
# EXÉCUTION DU PROGRAMME
# ==========================================
if __name__ == "__main__":
    # 1. Charger les JSON
    stats_data = json.loads(stats_json_str)
    schema_data = json.loads(schema_db1_json_str)
    
    # 2. Initialiser l'objet demandé pour le "Homework"
    simulator = DataModelSimulator(schema_data, stats_data)
    
    # 3. Répondre à la section 2.5 (Tailles)
    simulator.print_sizes_report()
    
    # 4. Répondre à la section 2.6 (Sharding)
    strategies_to_test = [
        ("Stock", "#IDP"),
        ("Stock", "#IDW"),
        ("OrderLine", "#IDC"),
        ("OrderLine", "#IDP"),
        ("Product", "#IDP"),
        ("Product", "#brand")
    ]
    simulator.print_sharding_report(strategies_to_test)
