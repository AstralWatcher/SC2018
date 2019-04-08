def onLine(line, p):
    if p[0] <= max(line[0], line[2]) and p[0] <= min(line[0], line[3]) and p[1] <= max(line[1], line[3]) and p[1] <= min(line[1], line[3]):
        return True
    return False

def direction(a, b, c):
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1]);
    if val == 0:
        return 0  # colinear
    elif val < 0:
        return 2  # anti-clockwise direction
    return 1  # clockwise direction


def is_intersect(l1, l2):
    # four direction for two lines and points of other line
    dir1 = direction([l1[0], l1[1]], [l1[2], l1[3]], [l2[0], l2[1]]);
    dir2 = direction([l1[0], l1[1]], [l1[2], l1[3]], [l2[2], l2[3]]);
    dir3 = direction([l2[0], l2[1]], [l2[2], l2[3]], [l1[0], l1[1]]);
    dir4 = direction([l2[0], l2[1]], [l2[2], l2[3]], [l1[2], l1[3]]);

    if dir1 != dir2 and dir3 != dir4:
        return True  # they are intersecting

    if dir1 == 0 and onLine(l1, [l2[0], l2[1]]):  # when p2 of line2 are on the line1
        return True

    if dir2 == 0 and onLine(l1, [l2[2], l2[3]]):  # when p1 of line2 are on the line1
        return True

    if dir3 == 0 and onLine(l2, [l1[0], l1[1]]):  # when p2 of line1 are on the line2
        return True

    if dir4 == 0 and onLine(l2, [l1[2], l1[3]]):  # when p1 of line1 are on the line2
        return True
    return False


def main():
    #l1 = [0, 0, 5, 5]
    #l2 = [10, 10, 3, 10]
    l1 = [1, 1, 2, 2]
    l2 = [1, 3, 3, 1]
    if is_intersect(l1, l2):
        print("Lines are intersecting")
    else:
        print("Lines are not intersecting")