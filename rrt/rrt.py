"""
测试RRT算法的实现Demo
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import random
import math
import copy
import numpy as np

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
        self.safe = Node(20, 20)
        self.safe.parent = 0
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.step = 10  # 探索步长
        self.distance = 5
        self.goalSampleRate = 0.05  # 选择终点的概率
        self.maxIter = 500
        self.obstacle_list = obstacle_list
        self.node_list = [self.start, self.safe]
        self.angle = 60

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
        """
        在节点列表中找到距离目标节点中最近的点
        :param node_list: 节点列表 T1 or obstacle_list
        :param rnd: 目标节点
        :return: 最近节点的位置
        """
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_nearest_obstacle_index(self, point):
        """
        找到距离point最近的障碍物
        :param point: 节点
        :return: 最近的障碍物
        """
        d_list = [(math.sqrt((point.x - x) ** 2 + (point.y - y) ** 2)) - r for (x, y, r) in self.obstacle_list]
        min_index = d_list.index(min(d_list))
        return min_index

    def collision_check(self, new_node, obstacle_list):
        """
        检查新的节点是否会碰撞
        :param new_node:  实新节点
        :param obstacle_list: 障碍物列表
        :return:
        """
        flag = 1
        for (ox, oy, radius) in obstacle_list:
            dx = ox - new_node.x
            dy = oy - new_node.y
            d = math.sqrt(dx * dx + dy * dy)

            parent_point = self.node_list[new_node.parent]
            vector_o = np.array([ox, oy])
            vector_p = np.array([parent_point.x, parent_point.y])
            vector_n = np.array([new_node.x, new_node.y])
            d_p_n = np.abs(np.cross(vector_p - vector_o, vector_n - vector_o)) / np.linalg.norm(vector_p - vector_n)

            if d <= radius or d_p_n < radius:
                flag = 0
                return flag
        return flag

    def dynamic_step(self, n_point, a_point):
        """
        计算动态步长
        :param n_point: 父节点
        :param a_point: 虚新节点
        :return: Sf 动态步长
        """

        tan = math.atan2(a_point.y - n_point.y, a_point.x - n_point.x)
        a_point.x += math.cos(tan) * (self.distance + self.step) / 2
        a_point.y += math.sin(tan) * (self.distance + self.step) / 2
        # 距离最近的障碍物
        obstacle_min = self.obstacle_list[self.get_nearest_obstacle_index(a_point)]
        # 虚拟节点a_point至障碍物边缘的距离l_a
        # l_a = math.sqrt((a_point.x - obstacle_min[0]) ** 2 + (a_point.y - obstacle_min[1]) ** 2) - obstacle_min[2]
        # 生长点n_point至障碍物边缘的距离l_n
        l_n = math.sqrt(np.square(n_point.x - obstacle_min[0]) + np.square(n_point.y - obstacle_min[1])) - obstacle_min[
            2]

        dynamic = self.step / (1 + (self.step / self.distance - 1) * math.exp(-3 * l_n / self.step))
        return dynamic

    def angle_check(self, new_point, parent_point, ancestor_point):
        """
        转角约束
        :param new_point: 新节点 w
        :param parent_point: n节点(新节点的父级节点)
        :param ancestor_point: f祖先节点
        :return:
        """
        vector_p_n = np.array([new_point.x - parent_point.x, new_point.y - parent_point.y])
        vector_a_p = np.array([parent_point.x - ancestor_point.x, parent_point.y - ancestor_point.y])
        angle = get_angle(vector_a_p, vector_p_n)
        if angle <= self.angle:
            return True
        else:
            return False

    def coordinate(self, rnd):
        """
        实际坐标计算
        :param rnd: 虚新节点
        :return: 实际新节点
        """
        # 在搜索树中找到距离rnd最近的节点
        min_index = self.get_nearest_list_index(self.node_list, rnd)
        nearest_point = self.node_list[min_index]

        # 按照原始步长计算坐标
        theta = math.atan2(rnd[1] - nearest_point.y, rnd[0] - nearest_point.x)
        new_point = copy.deepcopy(nearest_point)
        new_point.x += math.cos(theta) * self.step
        new_point.y += math.sin(theta) * self.step
        # 使用动态步长策略计算实际坐标
        actual_step = self.dynamic_step(nearest_point, new_point)
        new_point.x = nearest_point.x + math.cos(theta) * actual_step
        new_point.y = nearest_point.y + math.sin(theta) * actual_step
        new_point.parent = min_index

        return new_point

    def generate_path_list(self):
        path = [[self.goal.x, self.goal.y]]
        last_index = len(self.node_list) - 1
        while self.node_list[last_index].parent is not None:
            node = self.node_list[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def planning(self):
        """
        路径计划
        :return:
        """
        while True:
            # 生成随机点
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = [self.goal.x, self.goal.y]
            # 计算实新节点
            new_node = self.coordinate(rnd)
            # 碰撞检测
            if not self.collision_check(new_node, self.obstacle_list):
                continue
            # 父节点
            n_point = self.node_list[new_node.parent]
            if n_point.parent is None:
                continue
            # 祖先节点
            f_point = self.node_list[n_point.parent]
            # 转角约束
            if not self.angle_check(new_node, n_point, f_point):
                continue

            self.node_list.append(new_node)
            # 接近判断
            dx = new_node.x - self.goal.x
            dy = new_node.y - self.goal.y
            d = math.sqrt(dx * dx + dy * dy)
            if self.distance <= d <= self.step:
                print("goal")
                break

            # if True:
            #     self.draw_graph(rnd)
        # 路径回溯
        return self.generate_path_list()

    def draw_graph(self, rnd=None):
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

        x_major_location = MultipleLocator(10)
        # y轴刻度区间
        y_major_location = MultipleLocator(10)
        # 坐标轴实例
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_location)
        ax.yaxis.set_major_locator(y_major_location)

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


def get_angle(a, b):
    """
    向量夹角计算
    :param a: 向量1
    :param b: 向量2
    :return: 角度值
    """
    a_norm = np.sqrt(np.sum(a * a))
    b_norm = np.sqrt(np.sum(b * b))
    cos_value = float(np.dot(a, b) / (a_norm * b_norm))
    eps = 1e-6
    if 1.0 < cos_value < 1.0 + eps:
        cos_value = 1.0
    elif -1.0 - eps < cos_value < -1.0:
        cos_value = -1.0
    arc_value = np.arccos(cos_value)
    angle_value = arc_value * (180 / np.pi)
    return angle_value


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
