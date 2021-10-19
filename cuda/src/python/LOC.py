import sys
from typing import List

START_COMMENT: str = "// LOC.py start"
STOP_COMMENT: str = "// LOC.py stop"
IGNORE_COMMENT_START: str = "// LOC.py ignore start"
IGNORE_COMMENT_STOP: str = "// LOC.py ignore stop"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: LOC.py FILE_NAME [IDENTIFIER]")
        sys.exit(1)

    file_name: str = sys.argv[1]
    if len(sys.argv) == 3:
        identifier: str = sys.argv[2]
        START_COMMENT += f" {identifier}"
        STOP_COMMENT += f" {identifier}"

    print(file_name)
    with open(file_name, "r") as f:
        lines: List[str] = f.readlines()
        
        # remove whitespace
        lines = [l.strip() for l in lines]
        
        # remove empty strings
        lines = [l for l in lines if l]

        # lines = [l for l in lines if l != "}" and l != "};"]

        start: int = lines.index(START_COMMENT)
        stop: int = lines.index(STOP_COMMENT)
        lines = lines[start:stop]

        ignore_starts: List[int] = [i for i, line in enumerate(lines) if line == IGNORE_COMMENT_START]
        ignore_stops: List[int] = [i for i, line in enumerate(lines) if line == IGNORE_COMMENT_STOP]

        if len(ignore_starts) != len(ignore_stops):
            print(f"ERROR: {len(ignore_starts)} ignore starts and f{len(ignore_stops)} ignore stops")
            sys.exit(1)

        for i in ignore_starts:
            ignore_starts_temp: List[int] = [i for i, line in enumerate(lines) if line == IGNORE_COMMENT_START]
            ignore_stops_temp: List[int] = [i for i, line in enumerate(lines) if line == IGNORE_COMMENT_STOP]
            print(f"found on {ignore_starts_temp}", file=sys.stderr)
            print(f"found on {ignore_stops_temp}", file=sys.stderr)

            # for start, stop in zip(ignore_starts_temp, ignore_stops_temp):
            del lines[ignore_starts_temp[0]:ignore_stops_temp[0]]

        # remove comments
        lines = list(filter(lambda item: not (item.startswith("//") or item.startswith("/*") or item.startswith("*")), lines))

        LOC: int = len(lines)
        characters: int = sum(len(l) for l in lines)

        print(f"LOC: {LOC}")
        print(f"characters: {characters}")