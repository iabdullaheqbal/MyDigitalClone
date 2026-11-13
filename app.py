from clone import hf_runtime  # noqa: F401
from clone.bootstrap import create_app

demo = create_app()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()