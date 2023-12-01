from pydantic import Field


def Context(description: str):
    return Field(description=description)


if __name__ == "__main__":
    context = Context("hi")
    print(context)
