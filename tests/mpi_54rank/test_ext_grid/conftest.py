def pytest_addoption(parser):
    parser.addoption(
        "--config_file_path",
        action="store",
        default="config.yaml",
        help="Configuration for testing.",
    )
    parser.addoption(
        "--tile_file_base_path",
        action="store",
        default="tile.nc",
        help="Tile file base name to be read in",
    )
