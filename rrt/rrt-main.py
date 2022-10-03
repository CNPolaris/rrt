import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from birrt import Point, BiRRT


def draw_statistic(start, goal, min_rand, max_rand, obstacle_list, path):
    plt.clf()

    # 绘制区域
    # x轴刻度区间
    x_major_location = MultipleLocator(10)
    # y轴刻度区间
    y_major_location = MultipleLocator(10)
    # 坐标轴实例
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_location)
    ax.yaxis.set_major_locator(y_major_location)
    plt.axis([min_rand, max_rand, min_rand, max_rand])

    # 绘制起点
    plt.plot(start[0], start[1], "^g")
    # 绘制终点
    plt.plot(goal[0], goal[1], "^b")

    # 绘制障碍物
    for (ox, oy, radius) in obstacle_list:
        circle = plt.Circle(xy=(ox, oy), radius=radius, color="r")
        ax.add_patch(circle)
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot()

    # 绘制规划的路径
    plt.plot([data[0] for data in path], [data[1] for data in path], "-y")
    # for (x, y) in path:
    #     plt.scatter(x, y, marker='o', c='b', edgecolors='b')

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # 无人船的基本信息
    rand_area = [0, 100]  # 地图区域
    step = 10  # 基础步长
    angle = 60  # 最大转向角
    distance = 5  # 最小航行距离
    start = [0, 0]  # 起点
    goal = [100, 100]  # 终点
    safe = [20, 20]  # 安全航迹结束点
    recover = [90, 90]  # 安全航迹恢复点

    # 障碍物
    obstacle_list = [
        (50, 50, 15),
        (62, 13, 12),
        (50, 87, 11)
    ]
    rrt = BiRRT(start, goal, angle, step, distance, obstacle_list, rand_area, safe, recover)
    path = rrt.planning()
    draw_statistic(start, goal, rand_area[0], rand_area[1], obstacle_list, path)
