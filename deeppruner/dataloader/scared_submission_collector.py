from pathlib import Path

def datacollector(filepath):
    root_test_dir = Path(filepath).resolve()
    left_test = sorted([p for p in root_test_dir.rglob('left_rect/*.png')])
    right_test = sorted([p for p in root_test_dir.rglob('right_rect/*.png')])
    print(len(left_test))
    assert len(left_test) == len(right_test)
    return left_test, right_test
