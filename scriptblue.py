from typing import Optional, Union, Any
from collections import namedtuple
import random

name = "Avdhoot"

Point = namedtuple("Point", ["x", "y"])
Dimensions = namedtuple("Dimensions", ["w", "h"])


class PirateSignal:
    EXPLORE_CELL = "e"
    LAST_VISITED_CELL_MAP = "l"
    LAST_VISITED_ISLAND_MAP = "i"
    CAPTURE_ISLAND = "c"
    GUARD_ISLAND = "g"


class TeamSignal:
    UNEXPLORED_CELLS = "u"  # ids of the unexplored cells
    ISLAND_POS = "i"  # positions of the islands
    NUM_PIRATES_CAPTURE_ISLAND = "n"  # number of pirates to capture the islands
    NUM_PIRATES_GUARD_ISLAND = "m"  # number of pirates to capture the islands


class Constants:
    NUM_ISLANDS = 3
    SIGNAL_SEPARATOR = "|"
    CELL_EXPLORING_TIMESTAMP_LEN = 2
    ISLAND_SIZE = Dimensions(3, 3)
    NUM_MAX_EXPLORERS = Dimensions(8, 8)
    NUM_PIRATE_TO_CAPTURE = 5
    TIME_TO_START_HALF_CAPTURE = 1500  # Time after which half of all available pirates are sent to capture islands
    TIME_TO_START_FULL_CAPTURE = 1750  # Time after which all of the available pirates are sent to capture islands
    VALID_ENCODING_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*(){}[]_-+=:;,.<>/?`~"


class Helpers:
    @staticmethod
    def get_cell_size(obj) -> Dimensions:
        """Returns a tuple containing the size of the cell that the pirates should explore.
        Returns: (width, height)
        """

        if obj.getDimensionX() % Constants.NUM_MAX_EXPLORERS.w == 0:
            w = obj.getDimensionX() // Constants.NUM_MAX_EXPLORERS.w
        else:
            w = obj.getDimensionX() // (Constants.NUM_MAX_EXPLORERS[0] - 1)

        if obj.getDimensionY() % Constants.NUM_MAX_EXPLORERS.h == 0:
            h = obj.getDimensionY() // Constants.NUM_MAX_EXPLORERS.h
        else:
            h = obj.getDimensionY() // (Constants.NUM_MAX_EXPLORERS[1] - 1)

        return Dimensions(w, h)

    @staticmethod
    def find_signal(signals: Union[str, list[str]], signal_type: str) -> int:
        """Finds the index of the signal in the list of signals."""

        if isinstance(signals, str):
            signals = signals.split(Constants.SIGNAL_SEPARATOR)

        for i, signal in enumerate(signals):
            if signal.startswith(signal_type):
                return i

        return -1

    @staticmethod
    def get_signal_value(signal: str, signal_type: str) -> Optional[str]:
        """Finds the value of the signal in the signals of the object."""
        signals = signal.split(Constants.SIGNAL_SEPARATOR)
        signal_idx = Helpers.find_signal(signals, signal_type)
        if signal_idx == -1:
            return None

        return signals[signal_idx][len(signal_type) :]

    @staticmethod
    def update_signal(obj, signal_type: str, value: Optional[str]) -> None:
        signals = obj.getSignal().split(Constants.SIGNAL_SEPARATOR)
        signal_idx = Helpers.find_signal(signals, signal_type)
        if signal_idx == -1:
            if value is not None:
                signals.append(f"{signal_type}{value}")
        else:
            if value is None:
                signals.pop(signal_idx)
            else:
                signals[signal_idx] = f"{signal_type}{value}"

        obj.setSignal(
            Constants.SIGNAL_SEPARATOR.join(filter(lambda x: x != "", signals))
        )

    @staticmethod
    def update_team_signal(obj, signal_type: str, value: Optional[str]) -> None:
        signals = obj.getTeamSignal().split(Constants.SIGNAL_SEPARATOR)
        signal_idx = Helpers.find_signal(signals, signal_type)
        if signal_idx == -1:
            if value is not None:
                signals.append(f"{signal_type}{value}")
        else:
            if value is None:
                signals.pop(signal_idx)
            else:
                signals[signal_idx] = f"{signal_type}{value}"

        obj.setTeamSignal(
            Constants.SIGNAL_SEPARATOR.join(filter(lambda x: x != "", signals))
        )

    @staticmethod
    def encode_num_array(arr: list[int], padded_len_of_each: int) -> str:
        if len(arr) == 0:
            return ""

        # Add a '1' at the start to avoid leading zeros
        padded_num = "1" + "".join(
            [str.zfill(str(val), padded_len_of_each) for val in arr]
        )
        num_to_encode = int(padded_num)

        base = len(Constants.VALID_ENCODING_CHARS)
        encoded_num = ""
        while num_to_encode:
            encoded_num += Constants.VALID_ENCODING_CHARS[num_to_encode % base]
            num_to_encode //= base
        return encoded_num[::-1]

    @staticmethod
    def decode_num_array(encoded_str: str, padded_len_of_each: int) -> list[int]:
        decoded_num = 0
        base = len(Constants.VALID_ENCODING_CHARS)
        for char in encoded_str:
            decoded_num *= base
            decoded_num += Constants.VALID_ENCODING_CHARS.index(char)
        decoded_str = str(decoded_num)[1:]  # Remove the '1' that was added at the start
        return [
            int(decoded_str[i : i + padded_len_of_each])
            for i in range(0, len(decoded_str), padded_len_of_each)
        ]

    @staticmethod
    def _get_cell_loc(cell_id: int, deploy_cell: Point) -> Point:
        if cell_id == 0:
            return Point(deploy_cell.x, deploy_cell.y)

        _sum = 1
        X = 0
        Y = 0
        while True:
            for i in range(_sum + 1):
                for j in range(_sum + 1):
                    if (
                        i + j != _sum
                        or i >= Constants.NUM_MAX_EXPLORERS.w
                        or j >= Constants.NUM_MAX_EXPLORERS.h
                    ):
                        continue
                    X = i
                    Y = j
                    cell_id -= 1
                    if cell_id == 0:
                        # return Point(abs(X - deploy_cell.x), abs(Y - deploy_cell.y))
                        return Point(abs(X), abs(Y))
            _sum += 1

        # Row wise (OLD):
        # return (
        #     cell_id % Constants.NUM_MAX_EXPLORERS[0],
        #     cell_id // Constants.NUM_MAX_EXPLORERS[0],
        # )

    @staticmethod
    def get_deploy_point_cell_loc(deploy_point: Point, cell_size: Dimensions) -> Point:
        if deploy_point.x == 0 and deploy_point.y == 0:
            return Point(0, 0)
        if deploy_point.x == 0:
            return Point(0, deploy_point.y // cell_size.h)
        if deploy_point[1] == 0:
            return Point(deploy_point.x // cell_size.w, 0)
        return Point(deploy_point.x // cell_size.w, deploy_point.y // cell_size.h)

    @staticmethod
    def get_cell_center(
        cell_id: int, cell_size: Dimensions, deploy_cell: Point
    ) -> Point:
        (x, y) = Helpers._get_cell_loc(cell_id, deploy_cell)
        return Point(
            x * cell_size.w + cell_size.w // 2,
            y * cell_size.h + cell_size.h // 2,
        )

    @staticmethod
    def get_move_cmd(obj, x, y):
        position = obj.getPosition()
        if position[0] == x and position[1] == y:
            return 0
        if position[0] == x:
            return (position[1] < y) * 2 + 1
        if position[1] == y:
            return (position[0] > x) * 2 + 2
        delX = abs(position[0]-x)
        delY = abs(position[1]-y)
        if not (delX + delY):
            pos = 0
        else:
            pos = (delY)/((delX**2 +delY**2)**0.5)
        if random.randint(1, 2000) >= (pos)*1000:
            return (position[0] > x) * 2 + 2
        else:
            return (position[1] < y) * 2 + 1
    # def moveTo(x, y, Pirate):

    @staticmethod
    def is_obj_inside_cell(obj, cell_id: int) -> bool:
        dp = obj.getDeployPoint()
        dimX = obj.getDimensionX()
        dimY = obj.getDimensionY()
        # toMove = (dimX- dp[0]-1, dimY- dp[1])
        toMove = (dp[0], dp[1])
        cell_size = Helpers.get_cell_size(obj)
        cell_center = Helpers.get_cell_center(
            cell_id,
            cell_size,
            Helpers.get_deploy_point_cell_loc(
                Point(*toMove),
                cell_size,
            ),
        )
        position = Point(*obj.getPosition())

        x_ahead = cell_size.w // 2
        x_behind = cell_size.w // 2
        if cell_size.w % 2 == 0:
            x_ahead -= 1

        y_ahead = cell_size.h // 2
        y_behind = cell_size.h // 2
        if cell_size.h % 2 == 0:
            y_ahead -= 1

        return (
            position.x >= cell_center.x - x_behind
            and position.x <= cell_center.x + x_ahead
            and position.y >= cell_center.y - y_behind
            and position.y <= cell_center.y + y_ahead
        )

    @staticmethod
    def is_obj_inside_island(obj, island_center: Point) -> bool:
        position = Point(*obj.getPosition())
        return (
            position.x >= island_center.x - 1
            and position.x <= island_center.x + 1
            and position.y >= island_center.y - 1
            and position.y <= island_center.y + 1
        )

    @staticmethod
    def transform_global_position_to_cell(
        position: Point,
        cell_id: int,
        cell_size: Dimensions,
        deploy_cell: Point,
    ) -> Point:
        (x, y) = Helpers._get_cell_loc(cell_id, deploy_cell)
        return Point(
            position.x - x * cell_size.w,
            position.y - y * cell_size.h,
        )

    @staticmethod
    def transform_cell_position_to_global(
        position: Point,
        cell_id: int,
        cell_size: Dimensions,
        deploy_cell: Point,
    ) -> Point:
        (X, Y) = Helpers._get_cell_loc(cell_id, deploy_cell)
        return (
            position.x + X * cell_size.w,
            position.y + Y * cell_size.h,
        )

    @staticmethod
    def get_valid_positions(
        position_inside_cell: Point, cell_size: Dimensions
    ) -> list[Point]:
        return list(
            filter(
                lambda x: x is not None,
                [
                    (
                        Point(position_inside_cell.x + 1, position_inside_cell.y)
                        if position_inside_cell.x + 1 < cell_size.w
                        else None
                    ),
                    (
                        Point(position_inside_cell.x - 1, position_inside_cell.y)
                        if position_inside_cell.x - 1 >= 0
                        else None
                    ),
                    (
                        Point(position_inside_cell.x, position_inside_cell.y + 1)
                        if position_inside_cell.y + 1 < cell_size.h
                        else None
                    ),
                    (
                        Point(position_inside_cell.x, position_inside_cell.y - 1)
                        if position_inside_cell.y - 1 >= 0
                        else None
                    ),
                ],
            )
        )

    @staticmethod
    def flatten_2d(map: list[list[Any]]) -> list[Any]:
        flat_list = []
        for row in map:
            flat_list.extend(row)
        return flat_list

    @staticmethod
    def parse_island_positions(signal: str) -> list[Point]:
        if not signal:
            signal = "-,-,-"
        return [
            Point(int(x[:2]), int(x[2:])) if x != "-" else None
            for x in signal.split(",")
            if x != ""
        ]

    @staticmethod
    def update_island_positions(obj) -> None:
        island_positions = Helpers.parse_island_positions(
            Helpers.get_signal_value(
                obj.getTeamSignal(),
                TeamSignal.ISLAND_POS,
            )
        )
        # If all locations are known, return
        if (
            len(list(filter(lambda x: x is not None, island_positions)))
            == Constants.NUM_ISLANDS
        ):
            return

        up = obj.investigate_up()[0].startswith("island")
        down = obj.investigate_down()[0].startswith("island")
        left = obj.investigate_left()[0].startswith("island")
        right = obj.investigate_right()[0].startswith("island")
        nw = obj.investigate_nw()[0].startswith("island")
        ne = obj.investigate_ne()[0].startswith("island")
        sw = obj.investigate_sw()[0].startswith("island")
        se = obj.investigate_se()[0].startswith("island")
        x, y = obj.getPosition()

        # Update island positions
        # 1 -> Denotes current position
        # X -> Denotes island tile

        # Case 1:
        # 0 0 0 0 1
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, not down, not left, not right, not nw, not ne, sw, not se]):
            island_positions[int(obj.investigate_sw()[0][-1]) - 1] = Point(x - 2, y + 2)
        # Case 2:
        # 0 0 0 0 0
        # 0 X X X 1
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, not down, left, not right, not nw, not ne, sw, not se]):
            island_positions[int(obj.investigate_left()[0][-1]) - 1] = Point(
                x - 2, y + 1
            )
        # Case 3:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 1
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, not down, left, not right, nw, not ne, sw, not se]):
            island_positions[int(obj.investigate_left()[0][-1]) - 1] = Point(x - 2, y)
        # Case 4:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 1
        # 0 0 0 0 0
        if all([not up, not down, left, not right, nw, not ne, not sw, not se]):
            island_positions[int(obj.investigate_left()[0][-1]) - 1] = Point(
                x - 2, y - 1
            )
        # Case 5:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 1
        if all([not up, not down, not left, not right, nw, not ne, not sw, not se]):
            island_positions[int(obj.investigate_nw()[0][-1]) - 1] = Point(x - 2, y - 2)
        # Case 6:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 1 0
        if all([up, not down, not left, not right, nw, not ne, not sw, not se]):
            island_positions[int(obj.investigate_up()[0][-1]) - 1] = Point(x - 1, y - 2)
        # Case 7:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 1 0 0
        if all([up, not down, not left, not right, nw, ne, not sw, not se]):
            island_positions[int(obj.investigate_up()[0][-1]) - 1] = Point(x, y - 2)
        # Case 8:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 1 0 0 0
        if all([up, not down, not left, not right, not nw, ne, not sw, not se]):
            island_positions[int(obj.investigate_up()[0][-1]) - 1] = Point(x + 1, y - 2)
        # Case 9:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 1 0 0 0 0
        if all([not up, not down, not left, not right, not nw, ne, not sw, not se]):
            island_positions[int(obj.investigate_ne()[0][-1]) - 1] = Point(x + 2, y - 2)
        # Case 10:
        # 0 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 1 X X X 0
        # 0 0 0 0 0
        if all([not up, not down, not left, right, nw, not ne, not sw, not se]):
            island_positions[int(obj.investigate_right()[0][-1]) - 1] = Point(
                x + 2, y - 1
            )
        # Case 11:
        # 0 0 0 0 0
        # 0 X X X 0
        # 1 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, not down, not left, right, not nw, ne, not sw, se]):
            island_positions[int(obj.investigate_right()[0][-1]) - 1] = Point(x + 2, y)
        # Case 12:
        # 0 0 0 0 0
        # 1 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, not down, not left, right, not nw, not ne, not sw, se]):
            island_positions[int(obj.investigate_right()[0][-1]) - 1] = Point(
                x + 2, y + 1
            )
        # Case 13:
        # 1 0 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, not down, not left, not right, not nw, not ne, not sw, se]):
            island_positions[int(obj.investigate_se()[0][-1]) - 1] = Point(x + 2, y + 2)
        # Case 14:
        # 0 1 0 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, down, not left, not right, not nw, not ne, not sw, se]):
            island_positions[int(obj.investigate_down()[0][-1]) - 1] = Point(
                x + 1, y + 2
            )
        # Case 15:
        # 0 0 1 0 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, down, not left, not right, not nw, not ne, sw, se]):
            island_positions[int(obj.investigate_down()[0][-1]) - 1] = Point(x, y + 2)
        # Case 16:
        # 0 0 0 1 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 X X X 0
        # 0 0 0 0 0
        if all([not up, down, not left, not right, not nw, not ne, sw, not se]):
            island_positions[int(obj.investigate_down()[0][-1]) - 1] = Point(
                x - 1, y + 2
            )

        Helpers.update_team_signal(
            obj,
            TeamSignal.ISLAND_POS,
            ",".join(
                map(
                    lambda x: (
                        f"{str.zfill(str(x.x), 2)}{str.zfill(str(x.y), 2)}"
                        if x is not None
                        else "-"
                    ),
                    island_positions,
                )
            ),
        )

def ActPirate(pirate):
    pirate_signal = pirate.getSignal()
    team_signal = pirate.getTeamSignal()
    time = pirate.getCurrentFrame()
    position = Point(*pirate.getPosition())
    cell_size = Helpers.get_cell_size(pirate)
    dp = pirate.getDeployPoint()
    dimX = pirate.getDimensionX()
    dimY = pirate.getDimensionY()
    # toMove = (dimX- dp[0], dimY- dp[1])
    toMove = (dp[0], dp[1])
    print(f"Testing: ", toMove)
    deploy_cell = Helpers.get_deploy_point_cell_loc(Point(*toMove), cell_size)

    Helpers.update_island_positions(pirate)

    island_positions = Helpers.parse_island_positions(
        Helpers.get_signal_value(
            team_signal,
            TeamSignal.ISLAND_POS,
        )
    )

    # ------ Check if pirate is going to capture an island ------
    if (
        island_to_capture := Helpers.get_signal_value(
            pirate_signal, PirateSignal.CAPTURE_ISLAND
        )
    ) is not None:
        island_position = island_positions[int(island_to_capture)]

        # If pirate is not on the island, move towards the center of the island
        if not Helpers.is_obj_inside_island(pirate, island_position):
            return Helpers.get_move_cmd(pirate, *island_position)

        last_visited_island_map = Helpers.get_signal_value(
            pirate_signal,
            PirateSignal.LAST_VISITED_ISLAND_MAP,
        )
        if last_visited_island_map is None:
            last_visited_island_map = [0] * (
                Constants.ISLAND_SIZE.w * Constants.ISLAND_SIZE.h
            )
        else:
            last_visited_island_map = Helpers.decode_num_array(
                last_visited_island_map,
                Constants.CELL_EXPLORING_TIMESTAMP_LEN,
            )
        last_visited_island_map = [
            last_visited_island_map[i : i + Constants.ISLAND_SIZE.w]
            for i in range(0, len(last_visited_island_map), Constants.ISLAND_SIZE.w)
        ]

        position_inside_island = Point(
            position.x - (island_position.x - Constants.ISLAND_SIZE.w // 2),
            position.y - (island_position.y - Constants.ISLAND_SIZE.h // 2),
        )
        valid_positions = Helpers.get_valid_positions(
            position_inside_island, Constants.ISLAND_SIZE
        )
        min_val, min_pos = float("inf"), None
        for valid_position in valid_positions:
            if last_visited_island_map[valid_position.y][valid_position.x] < min_val:
                min_val = last_visited_island_map[valid_position.y][valid_position.x]
                min_pos = valid_position

        last_visited_island_map[min_pos.y][min_pos.x] = time - ((time // 100) * 100)

        if time % 100 == 0:
            for i in range(len(last_visited_island_map)):
                for j in range(len(last_visited_island_map[1])):
                    last_visited_island_map[i][j] = 0

        Helpers.update_signal(
            pirate,
            PirateSignal.LAST_VISITED_ISLAND_MAP,
            Helpers.encode_num_array(
                Helpers.flatten_2d(last_visited_island_map),
                Constants.CELL_EXPLORING_TIMESTAMP_LEN,
            ),
        )

        return Helpers.get_move_cmd(
            pirate,
            *Point(
                min_pos.x + (island_position.x - Constants.ISLAND_SIZE.w // 2),
                min_pos.y + (island_position.y - Constants.ISLAND_SIZE.h // 2),
            ),
        )

    # ------ Check if team wants pirates to capture any islands ------
    if (
        num_pirates_capture_islands := Helpers.get_signal_value(
            team_signal, TeamSignal.NUM_PIRATES_CAPTURE_ISLAND
        )
    ) is not None:
        num_pirates_capture_islands = list(
            map(int, num_pirates_capture_islands.split(","))
        )
        island_to_capture = None
        for i in range(Constants.NUM_ISLANDS):
            if num_pirates_capture_islands[i] > 0:
                island_to_capture = i
                break
        if island_to_capture is not None:
            island_position = island_positions[island_to_capture]
            Helpers.update_signal(
                pirate,
                PirateSignal.CAPTURE_ISLAND,
                str(island_to_capture),
            )
            num_pirates_capture_islands[island_to_capture] -= 1
            Helpers.update_team_signal(
                pirate,
                TeamSignal.NUM_PIRATES_CAPTURE_ISLAND,
                ",".join(map(str, num_pirates_capture_islands)),
            )
            Helpers.update_signal(pirate, PirateSignal.EXPLORE_CELL, None)
            Helpers.update_signal(
                pirate,
                PirateSignal.LAST_VISITED_CELL_MAP,
                None,
            )
            return Helpers.get_move_cmd(pirate, *island_position)

    # ------ Check if pirate is exploring a cell ------
    if (
        cell_id_to_explore := Helpers.get_signal_value(
            pirate_signal, PirateSignal.EXPLORE_CELL
        )
    ) is not None:
        cell_id_to_explore = int(cell_id_to_explore)

        # Check if the pirate is inside the cell
        if not Helpers.is_obj_inside_cell(pirate, cell_id_to_explore):
            # If not, move towards the center of the cell
            return Helpers.get_move_cmd(
                pirate,
                *Helpers.get_cell_center(cell_id_to_explore, cell_size, deploy_cell),
            )

        last_visited_cell_map = Helpers.get_signal_value(
            pirate_signal,
            PirateSignal.LAST_VISITED_CELL_MAP,
        )
        if last_visited_cell_map is None:
            last_visited_cell_map = [0] * (cell_size.w * cell_size.h)
        else:
            last_visited_cell_map = Helpers.decode_num_array(
                last_visited_cell_map,
                Constants.CELL_EXPLORING_TIMESTAMP_LEN,
            )
        last_visited_cell_map = [
            last_visited_cell_map[i : i + cell_size.w]
            for i in range(0, len(last_visited_cell_map), cell_size.w)
        ]

        position_inside_cell = Helpers.transform_global_position_to_cell(
            position,
            cell_id_to_explore,
            cell_size,
            deploy_cell,
        )
        valid_positions = Helpers.get_valid_positions(position_inside_cell, cell_size)
        min_val, min_pos = float("inf"), None
        for valid_position in valid_positions:
            if last_visited_cell_map[valid_position.y][valid_position.x] < min_val:
                min_val = last_visited_cell_map[valid_position.y][valid_position.x]
                min_pos = valid_position

        last_visited_cell_map[min_pos.y][min_pos.x] = time - ((time // 100) * 100)

        if time % 100 == 0:
            for i in range(len(last_visited_cell_map)):
                for j in range(len(last_visited_cell_map[1])):
                    last_visited_cell_map[i][j] = 0

        Helpers.update_signal(
            pirate,
            PirateSignal.LAST_VISITED_CELL_MAP,
            Helpers.encode_num_array(
                Helpers.flatten_2d(last_visited_cell_map),
                Constants.CELL_EXPLORING_TIMESTAMP_LEN,
            ),
        )

        return Helpers.get_move_cmd(
            pirate,
            *Helpers.transform_cell_position_to_global(
                min_pos,
                cell_id_to_explore,
                cell_size,
                deploy_cell,
            ),
        )
    else:
        # If not, check if there are unexplored cells
        unexplored_cells = Helpers.get_signal_value(
            team_signal,
            TeamSignal.UNEXPLORED_CELLS,
        )
        # If there are any unexplored cells, explore the closest one
        if len(unexplored_cells) > 0:
            unexplored_cells = Helpers.decode_num_array(unexplored_cells, 2)
            cell_id_to_explore = min(unexplored_cells)
            unexplored_cells.remove(cell_id_to_explore)
            Helpers.update_team_signal(
                pirate,
                TeamSignal.UNEXPLORED_CELLS,
                Helpers.encode_num_array(unexplored_cells, 2),
            )
            Helpers.update_signal(
                pirate,
                PirateSignal.EXPLORE_CELL,
                str.zfill(str(cell_id_to_explore), 2),
            )
            return Helpers.get_move_cmd(
                pirate,
                *Helpers.get_cell_center(cell_id_to_explore, cell_size, deploy_cell),
            )

    return random.randint(0, 4)


def ActTeam(team):
    time = team.getCurrentFrame()

    # Update unexplored cells
    num_pirates = team.getTotalPirates()
    pirate_signals = team.getListOfSignals()
    exploring_cells = set(
        map(
            lambda x: int(x),
            filter(
                lambda x: x is not None,
                [
                    Helpers.get_signal_value(signal, PirateSignal.EXPLORE_CELL)
                    for signal in pirate_signals
                ],
            ),
        )
    )
    unexplored_cells = [
        i
        for i in range(num_pirates)
        if i not in exploring_cells
        and i < Constants.NUM_MAX_EXPLORERS.w * Constants.NUM_MAX_EXPLORERS.h
    ]
    Helpers.update_team_signal(
        team,
        TeamSignal.UNEXPLORED_CELLS,
        Helpers.encode_num_array(unexplored_cells, 2),
    )

    island_statuses = team.trackPlayers()
    island_positions = Helpers.parse_island_positions(
        Helpers.get_signal_value(
            team.getTeamSignal(),
            TeamSignal.ISLAND_POS,
        )
    )
    num_pirates_capturing_islands = [0] * Constants.NUM_ISLANDS
    for val in [
        Helpers.get_signal_value(signal, PirateSignal.CAPTURE_ISLAND)
        for signal in pirate_signals
    ]:
        if val is not None:
            num_pirates_capturing_islands[int(val)] += 1
    num_pirates_capture_islands = Helpers.get_signal_value(
        team.getTeamSignal(),
        TeamSignal.NUM_PIRATES_CAPTURE_ISLAND,
    )
    if num_pirates_capture_islands is None:
        num_pirates_capture_islands = [0] * Constants.NUM_ISLANDS
    else:
        num_pirates_capture_islands = list(
            map(
                int,
                num_pirates_capture_islands.split(","),
            )
        )

    for i in range(Constants.NUM_ISLANDS):
        if num_pirates_capture_islands[i] == 0 and island_positions[i] is not None:
            if (
                island_statuses[i + Constants.NUM_ISLANDS] == "oppCapturing"
                and num_pirates_capturing_islands[i] < Constants.NUM_PIRATE_TO_CAPTURE
            ):
                num_pirates_capture_islands[i] = (
                    Constants.NUM_PIRATE_TO_CAPTURE - num_pirates_capturing_islands[i]
                )
            if time > Constants.TIME_TO_START_HALF_CAPTURE:
                num_pirates_capture_islands[i] = max(
                    (num_pirates // (2 * Constants.NUM_ISLANDS))
                    - num_pirates_capturing_islands[i],
                    0,
                )
            if time > Constants.TIME_TO_START_FULL_CAPTURE:
                num_pirates_capture_islands[i] = max(
                    (num_pirates // (Constants.NUM_ISLANDS))
                    - num_pirates_capturing_islands[i],
                    0,
                )

    Helpers.update_team_signal(
        team,
        TeamSignal.NUM_PIRATES_CAPTURE_ISLAND,
        ",".join(map(str, num_pirates_capture_islands)),
    )
