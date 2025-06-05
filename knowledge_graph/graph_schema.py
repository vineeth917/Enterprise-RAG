from neo4j import GraphDatabase

class GraphSchema:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.path IS UNIQUE")
            print("✅ Constraints created")

if __name__ == "__main__":
    db = GraphSchema("bolt://localhost:7687", "neo4j", "password")
    db.create_constraints()
    db.close()
