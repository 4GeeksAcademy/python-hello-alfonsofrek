import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app as inventory


class InventoryApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_file_patch = patch.object(
            inventory, "DATA_FILE", Path(self.temporary_directory.name) / "products.csv"
        )
        self.data_file_patch.start()
        self.client = TestClient(inventory.app)

    def tearDown(self):
        self.data_file_patch.stop()
        self.temporary_directory.cleanup()

    def test_create_update_alert_and_persist_product(self):
        created_response = self.client.post(
            "/inventory", json={"name": "Leche de avena", "quantity": 8, "unit": "litros"}
        )
        self.assertEqual(created_response.status_code, 201)
        product = created_response.json()

        updated_response = self.client.patch(
            f"/inventory/{product['id']}", json={"delta": 4}
        )
        self.assertEqual(updated_response.json()["quantity"], 12)
        self.assertEqual(self.client.get("/inventory").json()[0]["id"], product["id"])
        self.assertEqual(self.client.get("/inventory/alerts").json(), [])
        self.assertEqual(
            self.client.get("/inventory/alerts?threshold=13").json()[0]["id"],
            product["id"],
        )
        self.assertIn("Leche de avena", (Path(self.temporary_directory.name) / "products.csv").read_text())

    def test_rejects_negative_stock_and_unknown_product(self):
        created = self.client.post(
            "/inventory", json={"name": "Café", "quantity": 2, "unit": "kg"}
        ).json()
        negative_stock = self.client.patch(
            f"/inventory/{created['id']}", json={"delta": -3}
        )
        missing_product = self.client.patch("/inventory/no-such-id", json={"delta": 1})

        self.assertEqual(negative_stock.status_code, 422)
        self.assertEqual(missing_product.status_code, 404)


if __name__ == "__main__":
    unittest.main()