import csv
import os
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse


DATA_FILE = Path(os.getenv("INVENTORY_CSV_PATH", Path(__file__).resolve().parent.parent / "products.csv"))
FIELDS = ["id", "name", "quantity", "unit"]
app = FastAPI(title="API de inventario", version="1.0.0")


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse(url="/docs")


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    quantity: float = Field(ge=0)
    unit: str = Field(min_length=1, max_length=40)


class StockChange(BaseModel):
    delta: float


def read_products() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        return [
            {**row, "quantity": float(row["quantity"])}
            for row in csv.DictReader(file)
        ]


def write_products(products: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=DATA_FILE.parent, delete=False
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(products)
        temporary_path = Path(file.name)
    temporary_path.replace(DATA_FILE)


@app.get("/inventory")
def list_inventory() -> list[dict]:
    return read_products()


@app.post("/inventory", status_code=status.HTTP_201_CREATED)
def add_product(product: ProductCreate) -> dict:
    products = read_products()
    created = {"id": str(uuid4()), **product.model_dump()}
    products.append(created)
    write_products(products)
    return created


@app.patch("/inventory/{product_id}")
def change_stock(product_id: str, change: StockChange) -> dict:
    products = read_products()
    for product in products:
        if product["id"] == product_id:
            new_quantity = product["quantity"] + change.delta
            if new_quantity < 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="El stock no puede quedar por debajo de cero.",
                )
            product["quantity"] = new_quantity
            write_products(products)
            return product
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Producto no encontrado.")


@app.get("/inventory/alerts")
def inventory_alerts(
    threshold: Annotated[float, Query(ge=0)] = 10,
) -> list[dict]:
    return [product for product in read_products() if product["quantity"] < threshold]