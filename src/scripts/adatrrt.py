#!/usr/bin/env python
import adapy
import rospy
import numpy as np
import time

use_tRRT = True
use_APFRRT = False


class AdatRRT():
    """
    Rapidly-Exploring Random Trees (RRT) for the ADA controller.
    """
    joint_lower_limits = np.array([-3.14, 1.57, 0.33, -3.14, 0, 0])
    joint_upper_limits = np.array([3.14, 5.00, 5.00, 3.14, 3.14, 3.14])

    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            """
            :param state: np.array of a state in the search space.
            :param parent: parent Node object.
            """
            self.state = np.asarray(state)  # state.copy()
            self.parent = parent
            self.children = []

        def __iter__(self):
            """
            Breadth-first iterator.
            """
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            """
            Adds a new child at the given state.

            :param state: np.array of new child node's state
            :returns: child Node object.
            """
            child = AdatRRT.Node(state=state, parent=self)
            self.children.append(child)
            return child

    def __init__(self,
                 start_state,
                 goal_state,
                 ada,
                 joint_lower_limits=None,
                 joint_upper_limits=None,
                 ada_collision_constraint=None,
                 step_size=0.25,
                 goal_precision=1.0,
                 rho_r=0.5,
                 rho_t=0.5,
                 obstacles=[],
                 max_iter=10000):
        """
        :param start_state: Array representing the starting state.
        :param goal_state: Array representing the goal state.
        :param ada: libADA instance.
        :param joint_lower_limits: List of lower bounds of each joint.
        :param joint_upper_limits: List of upper bounds of each joint.
        :param ada_collision_constraint: Collision constraint object.
        :param step_size: Distance between nodes in the RRT.
        :param goal_precision: Maximum distance between RRT and goal before
            declaring completion.
        :param sample_near_goal_prob:
        :param sample_near_goal_range:
        :param max_iter: Maximum number of iterations to run the RRT before
            failure.
        """
        self.start = AdatRRT.Node(start_state, None)
        self.goal = AdatRRT.Node(goal_state, None)
        self.ada = ada
        self.joint_lower_limits = joint_lower_limits or AdatRRT.joint_lower_limits
        self.joint_upper_limits = joint_upper_limits or AdatRRT.joint_upper_limits
        self.ada_collision_constraint = ada_collision_constraint
        self.step_size = step_size
        self.goal_precision = goal_precision
        self.rho_r = rho_r
        self.rho_t = rho_t
        self.obstacles = obstacles
        self.max_iter = max_iter

    def add_target_gravity(self, neighbor, sample):
        goal_dist = np.linalg.norm(neighbor.state - self.goal.state)
        rand_dist = np.linalg.norm(neighbor.state - sample)

        sample = neighbor.state + (self.rho_r * (sample - neighbor.state) / rand_dist + self.rho_t * (self.goal.state - neighbor.state) / goal_dist) * rand_dist
        return sample

    def build(self):
        """
        Build an RRT.
        In each step of the RRT:
            1. Sample a random point.
            2. Find its nearest neighbor.
            3. Attempt to create a new node in the direction of sample from its nearest neighbor.
            4. If we have created a new node, check for completion.
        Once the RRT is complete, add the goal node to the RRT and build a path from start to goal.

        :returns: A list of states that create a path from start to goal on success. On failure, returns None.
        """
        for k in range(self.max_iter):
            if np.random.random_sample(1) > 0.8:
                sample = self._get_random_sample()
            else:
                sample = self._get_random_state_near_goal()
            
            neighbor = self._get_nearest_neighbor(sample)

            if use_tRRT:
                print("Using tRRT")
                # Offset new random sample with target gravity
                sample = self.add_target_gravity(neighbor, sample)

                # get new neighbor for new sample config
                neighbor = self._get_nearest_neighbor(sample)

            # too close
            if np.linalg.norm(neighbor.state - sample) < self.step_size:
                continue

            new_node = self._extend_sample(sample, neighbor)

            if new_node and self._check_for_completion(new_node):
                self.goal.parent = new_node
                path = self._trace_path_from_start()
                return path

        print("Failed to find path from {0} to {1} after {2} iterations!".format(
            self.start.state, self.goal.state, self.max_iter))
        return None

    def _get_random_sample(self):
        """
        Uniformly samples the search space.

        :returns: A vector representing a randomly sampled point in the search space.
        """
        random_sample = np.array([])
        for bound in range(len(self.joint_lower_limits)):
            random_sample = np.append(random_sample, np.random.uniform(low=self.joint_lower_limits[bound],
                                                                       high=self.joint_upper_limits[bound]))
        return random_sample

    def _get_random_state_near_goal(self):
        state = np.random.random_sample(self.joint_lower_limits.size)
        state = state * self.step_size
        state = self.goal.state + state
        return state

    def _get_nearest_neighbor(self, sample):
        """
        Finds the closest node to the given sample in the search space,
        excluding the goal node.

        :param sample: The target point to find the closest neighbor to.
        :returns: A Node object for the closest neighbor.
        """
        closest_node = self.start
        for node in self.start:
            if np.linalg.norm(sample - closest_node.state) > np.linalg.norm(sample - node.state):
                closest_node = node
        return closest_node

    def _extend_sample(self, sample, neighbor):
        """
        Adds a new node to the RRT between neighbor and sample, at a distance
        step_size away from neighbor. The new node is only created if it will
        not collide with any of the collision objects (see
        RRT._check_for_collision)

        :param sample: target point
        :param neighbor: closest existing node to sample
        :returns: The new Node object. On failure (collision), returns None.
        """
        direction = (sample - neighbor.state) / np.linalg.norm(sample - neighbor.state)
        new_state = neighbor.state + (direction * self.step_size)
        if self._check_for_collision(new_state):
            return None
        else:
            child = neighbor.add_child(new_state)
            return child

    def _check_for_completion(self, node):
        """
        Check whether node is within self.goal_precision distance of the goal.

        :param node: The target Node
        :returns: Boolean indicating node is close enough for completion.
        """
        return np.linalg.norm(node.state - self.goal.state) < self.goal_precision

    def _trace_path_from_start(self, node=None):
        """
        Traces a path from start to node, if provided, or the goal otherwise.

        :param node: The target Node at the end of the path. Defaults to self.goal
        :returns: A list of states (not Nodes!) beginning at the start state and ending at the goal state.
        """
        node = node if node else self.goal
        state_list = [node.state]
        while (node is not self.start) and node.parent:
            node = node.parent
            state_list.insert(0, node.state)
        return state_list

    def _check_for_collision(self, new_node_state):
        """
        Checks if a sample point is in collision with any collision object.

        :returns: A boolean value indicating that sample is in collision. False means no collision.
        """
        print("Distance: ", self.ada.get_distance_to_obstacle(self.ada.get_arm_state_space(), self.ada.get_arm_skeleton(), self.obstacles, new_node_state))

        print("Collision: ", not self.ada_collision_constraint.is_satisfied(self.ada.get_arm_state_space(),
                                                          self.ada.get_arm_skeleton(), new_node_state))

        if self.ada_collision_constraint is None:
            return False
        return not self.ada_collision_constraint.is_satisfied(self.ada.get_arm_state_space(),
                                                          self.ada.get_arm_skeleton(), new_node_state)


def main():
    sim = True

    # instantiate an ada
    ada = adapy.Ada(True)

    armHome = [-1.5, 3.22, 1.23, -2.19, 1.8, 1.2]
    goalConfig = [-1.72, 4.44, 2.02, -2.04, 2.66, 1.39]
    delta = 0.1
    eps = 0.1
    rho_r = 0.6
    rho_t = 1.0 - rho_r

    D_THETA = 5
    D_DIST = 0.2

    ATTRACTIVE_GAIN = 10000.0
    REPULSIVE_GAIN = 15.0


    if sim:
        ada.set_positions(armHome)

    # launch viewer
    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    # add objects to world
    canURDFUri = "package://pr_assets/data/objects/can.urdf"
    sodaCanPose = [0.25, -0.35, 0.0, 0, 0, 0, 1]
    tableURDFUri = "package://pr_assets/data/furniture/uw_demo_table.urdf"
    tablePose = [0.3, 0.0, -0.0, 0.707107, 0, 0, 0.707107]
    world = ada.get_world()
    can = world.add_body_from_urdf(canURDFUri, sodaCanPose)
    # table = world.add_body_from_urdf(tableURDFUri, tablePose)

    boxURDFUri = "package://pr_assets/data/objects/box.urdf"
    boxPose = [0.28, -0.4, 0.3, 0, 0, 0, 1]
    box = world.add_body_from_urdf(boxURDFUri, boxPose)

    # add collision constraints
    collision_free_constraint = ada.set_up_collision_detection(ada.get_arm_state_space(),
                                                               ada.get_arm_skeleton(), [can, box])
    full_collision_constraint = ada.get_full_collision_constraint(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            collision_free_constraint)

    # easy goal
    adatRRT = AdatRRT(
        start_state=np.array(armHome),
        goal_state=np.array(goalConfig),
        ada=ada,
        ada_collision_constraint=full_collision_constraint,
        step_size=delta,
        goal_precision=eps,
        rho_r = rho_r,
        rho_t = rho_t,
        obstacles = [can, box])

    rospy.sleep(1.0)

    path = adatRRT.build()
    if path is not None:
        print("Path waypoints:")
        print(np.asarray(path))
        waypoints = []
        for i, waypoint in enumerate(path):
            waypoints.append((0.0 + i, waypoint))

        traj = ada.compute_joint_space_path(ada.get_arm_state_space(), waypoints)
        print("Path obstacle clear distance:")
        for i, waypoint in enumerate(path):
            print(adatRRT.ada.get_distance_to_obstacle(adatRRT.ada.get_arm_state_space(), adatRRT.ada.get_arm_skeleton(), adatRRT.obstacles, waypoint))
        
        raw_input('Press ENTER to execute trajectory and exit')
        ada.execute_trajectory(traj)
        rospy.sleep(10.0)


if __name__ == '__main__':
    main()
