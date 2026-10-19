import itertools
import json
import os
import sys
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from multiprocessing import Manager
from pathlib import Path
from typing import Annotated, Any, Iterable, Sized

from icecream import ic
from loguru import logger

ic.enable()
os.chdir(ic(Path(__file__).parents[2]))  # 切换到项目根目录作为工作目录
sys.path.append(".")

Check_Object = Enum("Check_Object", ["dt", "one_station", "all_station", "telescope"])


class Searcher:
    def __init__(self, query_dt: str | None = None, telescope=None) -> None:
        self.query_dt = query_dt
        self.telescope = telescope
        self.dt_path: Path = Path(f"data/{ic(self.query_dt)}")
        assert self.validate_path_exists(Check_Object.dt, self.dt_path, None)
        self.path_list: list[Path] = list(self.dt_path.iterdir())
        ic(self.path_list)

    def validate_path_exists(
        self,
        check_object: Check_Object,
        path: Path | None,
        path_iterator: Sized | None = None,
    ) -> bool:
        try:
            if check_object.name == "dt" and path:
                assert path.is_dir()
            elif check_object.name == "telescope" and path:
                assert path.is_file()
            elif check_object.name == "one_station" and path_iterator is not None:
                assert len(path_iterator) == 1
            elif check_object.name == "all_station" and path_iterator is not None:
                assert len(path_iterator) != 0
            else:
                raise AssertionError
        except:
            err_msg = f"❌ Path {str(path)} is invalid when checking parametre {check_object.name}. Either this is an invalid param or OB list with this param not generated yet."
            logger.info(err_msg)
            raise FileExistsError(err_msg)
        else:
            return True

    def find_all_station(self):
        all_dirs: list[Path] = list(
            filter(lambda x: x.name.startswith("output_"), self.path_list)
        )
        ic(all_dirs)
        assert self.validate_path_exists(Check_Object.all_station, None, all_dirs)
        all_dirs_and_station: list[tuple[Path, str]] = list(
            zip(all_dirs, list(map(lambda x: x.name.split("_")[1], all_dirs)))
        )
        ic(all_dirs_and_station)

        json_files_list = []
        for dir, dir_station in all_dirs_and_station:
            json_files: Iterable = filter(
                lambda x: str(x).endswith(".json"), ic(dir).iterdir()
            )
            json_files_list.append((json_files, dir_station))

        return json_files_list

    def match_pattern(self, psedu_file_path):
        path_exist: Iterable = map(
            lambda x: (
                True if str(x).lower() == str(psedu_file_path).lower() else False
            ),
            self.path_list,
        )
        target_path: list[Path] = list(itertools.compress(self.path_list, path_exist))
        ic(target_path)
        return target_path

    def find_one_station(self, station: str) -> Iterable[Path]:
        psedu_file_path: Path = self.dt_path / f"output_{station}"
        ic(psedu_file_path)

        candidate_path = self.match_pattern(psedu_file_path)

        assert self.validate_path_exists(Check_Object.one_station, None, candidate_path)
        station_path = candidate_path[0]
        if self.telescope:
            telescope_path: Path = station_path / f"{str(self.telescope)}.ninaTargetSet"
            assert self.validate_path_exists(
                Check_Object.telescope, telescope_path, None
            )
            return [telescope_path]
        else:
            json_files: Iterable = filter(
                lambda x: str(x).endswith(".json"), station_path.iterdir()
            )
            return json_files
