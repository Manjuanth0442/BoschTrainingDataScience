import json

from serve_model_api import app


def main() -> None:
    openapi_schema = app.openapi()
    with open("swagger.json", "w", encoding="utf-8") as file:
        json.dump(openapi_schema, file, indent=2)
    print("Saved Swagger document: swagger.json")


if __name__ == "__main__":
    main()
