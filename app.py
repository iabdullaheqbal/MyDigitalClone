from clone.bootstrap import create_app

if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=4).launch()