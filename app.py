import os

from clone.bootstrap import create_app

if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )