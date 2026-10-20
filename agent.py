import csv
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
API_URL = os.getenv("INVENTORY_API_URL", "http://127.0.0.1:8000")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LOG_FILE = Path(os.getenv("CONVERSATION_LOG_PATH", "conversation_log.csv"))
MAX_TOOL_ROUNDS = 8

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_inventory",
            "description": "Consulta el inventario completo. Úsala antes de modificar un producto si no conoces su identificador.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_product",
            "description": "Registra un producto nuevo en el inventario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Nombre del producto"},
                    "quantity": {"type": "number", "description": "Cantidad inicial"},
                    "unit": {"type": "string", "description": "Unidad, por ejemplo bolsas, kg o litros"},
                },
                "required": ["name", "quantity", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_stock",
            "description": "Suma una entrada o registra una salida de stock. Usa delta positivo para entradas y negativo para ventas. Busca primero el producto por nombre en el inventario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Nombre del producto existente"},
                    "delta": {"type": "number", "description": "Cambio de cantidad: positivo para entrada, negativo para salida"},
                },
                "required": ["product_name", "delta"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inventory_alerts",
            "description": "Devuelve los productos por debajo del umbral de stock indicado (por defecto, 10).",
            "parameters": {
                "type": "object",
                "properties": {"threshold": {"type": "number", "minimum": 0}},
                "additionalProperties": False,
            },
        },
    },
]


def log_event(actor: str, message: str, tool_call: str = "") -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["actor", "message", "tool_call", "timestamp"])
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "actor": actor,
                "message": message,
                "tool_call": tool_call,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


def api_request(method: str, path: str, payload: dict | None = None) -> dict | list:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{API_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8")
        raise RuntimeError(f"La API respondió {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"No se pudo conectar con la API en {API_URL}: {error.reason}") from error


def execute_tool(name: str, arguments: dict) -> dict | list:
    if name == "list_inventory":
        return api_request("GET", "/inventory")
    if name == "add_product":
        return api_request("POST", "/inventory", arguments)
    if name == "inventory_alerts":
        threshold = arguments.get("threshold", 10)
        return api_request("GET", f"/inventory/alerts?threshold={threshold}")
    if name == "change_stock":
        products = api_request("GET", "/inventory")
        requested_name = arguments["product_name"].strip().casefold()
        matches = [
            product for product in products
            if product["name"].strip().casefold() == requested_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Se encontraron {len(matches)} productos con el nombre "
                f"'{arguments['product_name']}'. Consulta el inventario y aclara el nombre."
            )
        product = matches[0]
        return api_request(
            "PATCH",
            f"/inventory/{product['id']}",
            {"delta": arguments["delta"]},
        )
    raise ValueError(f"Tool desconocida: {name}")


def run_turn(user_text: str, client: OpenAI, messages: list[dict]) -> str:
    messages.append({"role": "user", "content": user_text})
    log_event("user", user_text)

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))
        if not assistant_message.tool_calls:
            answer = assistant_message.content or "No tengo una respuesta para esa solicitud."
            log_event("agent", answer)
            return answer

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments or "{}")
            log_event("tool", json.dumps(arguments, ensure_ascii=False), name)
            try:
                result = execute_tool(name, arguments)
                result_text = json.dumps(result, ensure_ascii=False)
            except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
                result_text = json.dumps({"error": str(error)}, ensure_ascii=False)
            log_event("tool", result_text, name)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
            )

    answer = "No pude completar la solicitud porque se alcanzó el límite de pasos del agente."
    messages.append({"role": "assistant", "content": answer})
    log_event("agent", answer)
    return answer


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Falta GROQ_API_KEY. Configúrala en el fichero .env antes de iniciar el agente.")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    messages = [
        {
            "role": "system",
            "content": (
                "Eres el asistente de inventario de una tienda. Responde en español, "
                "con claridad y brevedad. Usa las herramientas para consultar o cambiar "
                "datos; nunca afirmes que una operación se completó si la API dio error. "
                "Para cambiar existencias, primero consulta el inventario y usa el nombre exacto."
            ),
        }
    ]
    print("Asistente de inventario listo. Escribe 'salir' para terminar.")
    while True:
        try:
            user_text = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSesión finalizada.")
            break
        if user_text.casefold() in {"salir", "exit", "quit"}:
            break
        if not user_text:
            continue
        try:
            print(f"Agente: {run_turn(user_text, client, messages)}")
        except Exception as error:
            print(f"Agente: No pude procesar la solicitud: {error}")


if __name__ == "__main__":
    main()