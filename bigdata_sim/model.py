"""
Homework 2.7 : structure d'une base NoSQL à partir d'un JSON Schema + statistiques,
calcul des tailles (document, collection, base) et de la distribution par sharding.
"""
import json

# Tailles en octets (section 2.5.1)
TYPE_SIZES = {
    "integer": 8,
    "number": 8,
    "string": 80,
    "date": 20,
    "longstring": 200,
}
KEY_SIZE = 12  # "Key+Value pairs/Arrays : 12B + values"
GB = 10 ** 9


# ==========================================
# STRUCTURE (objets réutilisés dans les TD suivants)
# ==========================================

class Field:
    """Une clé d'un document : type primitif, objet imbriqué ou tableau."""

    def __init__(self, name, kind, children=None, item=None):
        self.name = name
        self.kind = kind                # "integer", "string", "date", ..., "object" ou "array"
        self.children = children or []  # sous-clés si kind == "object"
        self.item = item                # type des éléments si kind == "array"


class Collection:
    def __init__(self, name, fields, nb_docs):
        self.name = name
        self.fields = fields
        self.nb_docs = nb_docs


class Database:
    def __init__(self, name, collections, stats):
        self.name = name
        self.collections = collections  # dict : nom -> Collection
        self.stats = stats

    # ---------- Tailles (2.5) ----------

    def avg_array_length(self, array_name, parent_name):
        """Nombre moyen d'éléments d'un tableau imbriqué.
        Valeur donnée dans les stats (ex. Categories = 2), sinon
        nb(enfant) / nb(parent) (ex. Stock dans Product = 2.10^7 / 10^5 = 200)."""
        if array_name in self.stats["array_lengths"]:
            return self.stats["array_lengths"][array_name]
        nb_docs = self.stats["nb_docs"]
        return nb_docs[array_name] / nb_docs[parent_name]

    def value_size(self, field, parent_name):
        """Taille de la valeur seule (sans la clé)."""
        if field.kind == "object":
            # Un objet qui correspond à une entité devient le parent de ses tableaux
            if field.name in self.stats["nb_docs"]:
                parent_name = field.name
            return sum(self.field_size(c, parent_name) for c in field.children)
        if field.kind == "array":
            n = self.avg_array_length(field.name, parent_name)
            return n * self.value_size(field.item, parent_name)
        return TYPE_SIZES[field.kind]

    def field_size(self, field, parent_name):
        """Taille d'une clé = 12B + taille de sa valeur."""
        return KEY_SIZE + self.value_size(field, parent_name)

    def document_size(self, coll_name):
        """Taille moyenne d'un document en octets."""
        coll = self.collections[coll_name]
        return sum(self.field_size(f, coll_name) for f in coll.fields)

    def collection_size(self, coll_name):
        """Taille d'une collection en octets."""
        return self.document_size(coll_name) * self.collections[coll_name].nb_docs

    def database_size(self):
        """Taille de la base en octets."""
        return sum(self.collection_size(name) for name in self.collections)

    # ---------- Sharding (2.6) ----------

    def sharding_stats(self, coll_name, key):
        """Distribution des documents de `coll_name` sur les serveurs avec la clé `key`.
        S'il y a moins de valeurs distinctes que de serveurs, seuls
        `distinct` serveurs reçoivent des données (les autres restent vides)."""
        coll = self.collections[coll_name]
        if find_field(coll.fields, key) is None:
            raise KeyError(f"La clé '{key}' n'existe pas dans {coll_name} ({self.name})")
        servers = self.stats["servers"]
        distinct = self.stats["distinct_values"][key]
        used_servers = min(servers, distinct)
        return {
            "used_servers": used_servers,
            "docs_per_server": coll.nb_docs / used_servers,
            "distinct_per_server": distinct / used_servers,
        }


def find_field(fields, name):
    """Cherche une clé, y compris dans les objets imbriqués (pas dans les tableaux :
    on ne peut pas sharder sur une clé contenue dans un tableau)."""
    for f in fields:
        if f.name == name:
            return f
        if f.kind == "object":
            found = find_field(f.children, name)
            if found:
                return found
    return None


# ==========================================
# CHARGEMENT : JSON Schema + stats -> Database
# ==========================================

def parse_field(name, schema):
    kind = schema["type"]
    if kind == "object":
        children = [parse_field(k, v) for k, v in schema["properties"].items()]
        return Field(name, "object", children=children)
    if kind == "array":
        return Field(name, "array", item=parse_field(name, schema["items"]))
    if kind == "string" and "format" in schema:
        return Field(name, schema["format"])  # "date" ou "longstring"
    return Field(name, kind)


def load_database(schema_path, stats_path):
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    collections = {}
    for coll_name, coll_schema in schema["properties"].items():
        fields = parse_field(coll_name, coll_schema).children
        collections[coll_name] = Collection(coll_name, fields, stats["nb_docs"][coll_name])
    return Database(schema.get("title", schema_path), collections, stats)
