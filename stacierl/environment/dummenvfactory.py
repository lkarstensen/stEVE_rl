from . import EnvFactory, Environment, DummyyEnv


class DummyEnvFactory(EnvFactory):
    def __init__(self) -> None:
        ...

    def create_env(self) -> Environment:
        return DummyyEnv()
