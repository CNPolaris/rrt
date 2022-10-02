"""
测试RRT算法的实现Demo
"""

import matplotlib.pyplot as plt
import random
import math
import copy

show = True


class Node(object):
    """
    节点
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


class RRT(object):
    """
    RRT 绘制
    """

    def __init__(self, start, goal, obstacle_list, rand_area):
        """
        :param start: [x,y]
        :param goal: [x,y]
        :param obstacle_list: [[x,y,size]...]
        :param rand_area: [min,max]
        :return:
        """
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expandDis = 5  # 探索步长
        self.goalSampleRate = 0.05  # 选择终点的概率
        self.maxIter = 500
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]

    def random_node(self):
        """
        产生随机节点
        :return:
        """
        node_x = random.uniform(self.min_rand, self.max_rand)
        node_y = random.uniform(self.min_rand, self.max_rand)
        node = [node_x, node_y]
        return node

    @staticmethod
    def get_nearest_list_index(node_list, rnd):
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        min_index = d_list.index(min(d_list))
        return min_index

    @staticmethod
    def collision_check(new_node, obstacle_list):
        """
        检查新的节点是否会碰撞
        :param new_node:
        :param obstacle_list:
        :return:
        """
        a = 1
        for (ox, oy, size) in obstacle_list:
            dx = ox - new_node.x
            dy = oy - new_node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= size:
                a = 0
        return a

    def planning(self):
        """
        路径计划
        :return:
        """
        while True:
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = [self.goal.x, self.goal.y]

            min_index = self.get_nearest_list_index(self.node_list, rnd)

            nearest_node = self.node_list[min_index]

            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
            new_node = copy.deepcopy(nearest_node)
            new_node.x += self.expandDis * math.cos(theta)
            new_node.y += self.expandDis * math.sin(theta)
            new_node.parent = min_index

            if not self.collision_check(new_node, self.obstacle_list):
                continue

            self.node_list.append(new_node)

            dx = new_node.x - self.goal.x
            dy = new_node.y - self.goal.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
                print("goal")
                break

            # if True:
            #     self.draw_graph(rnd)

        path = [[self.goal.x, self.goal.y]]
        last_index = len(self.node_list) - 1
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])

        return path

    def draw_graph(self, rnd=None):
        print("find")
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^g")
        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.x, self.node_list[node.parent].x],
                         [node.y, self.node_list[node.parent].y],
                         "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ko", ms=size)

        plt.plot(self.start.x, self.start.y, "^r")
        plt.plot(self.goal.x, self.goal.y, "^b")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def draw_static(self, path):
        """
        绘制静态图像
        :param path: 规划出的路径 [[x,y]...]
        :return:
        """
        plt.clf()

        for node in self.node_list:
            if node.parent is not None:
                plt.plot([node.x, self.node_list[node.parent].x],
                         [node.y, self.node_list[node.parent].y],
                         "-g")
        # 绘制障碍物
        ax = plt.gca()

        for (ox, oy, size) in self.obstacle_list:
            circle = plt.Circle(xy=(ox, oy), radius=size, color="r")
            ax.add_patch(circle)
        ax.set_aspect('equal', adjustable='datalim')
        ax.plot()
        # 绘制起点
        plt.plot(self.start.x, self.start.y, "^r")
        # 绘制终点
        plt.plot(self.goal.x, self.goal.y, "^b")
        # 绘制方框
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        # 绘制规划的路线
        plt.plot([data[0] for data in path], [data[1] for data in path], "-r")
        for (px, py) in path:
            plt.plot(px, py, '-r')
        plt.grid(True)
        plt.show()


def main():
    print("start RRT path planning")
    print("===============================================")

    obstacle_list = [
        (50, 50, 15),
        (62, 13, 12),
        (50, 87, 11)
    ]
    rrt = RRT(start=[0, 0], goal=[100, 100], rand_area=[-2, 100], obstacle_list=obstacle_list)
    path = rrt.planning()
    print(path)

    if show:
        plt.close()
        rrt.draw_static(path)

    print("===============================================")
    print("end")


if __name__ == '__main__':
    main()
