# -*- coding: utf-8 -*-
__author__ = "William Sena <@wllsena>"
"""
Style Guide: PEP 8. Column limit: 100.
Author: William Sena <@wllsena>.
"""

import uuid
from multiprocessing import Lock
from multiprocessing.pool import ThreadPool
from random import choice, randint
from time import sleep
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph

# -----

# (index, on_street, velocity, odometer)
car_result = Tuple[int, int, int, int]

# (index, steps, n_crossing, n_streets, n_cars, car_results)
city_result = Tuple[str, int, int, int, int, List[car_result]]


class Crossing:
    index: int

    n_streets: int
    streets: List[int]

    #butler
    #streets_lock

    def __init__(self, index: int):
        self.index = index

        self.n_streets = 0
        self.streets = []
        self.butler = Lock()
        self.streets_lock = []

    def add_street(self, street: int) -> None:
        self.n_streets += 1
        self.streets.append(street)
        self.streets_lock.append(Lock())

    def to_cross(self, street: int, new_street: int, delay: float = 0) -> None:
        if new_street == street:
            with self.butler:
                self.streets_lock[self.streets.index(street)].acquire()

            self.streets_lock[self.streets.index(street)].release()

        else:
            with self.butler:
                self.streets_lock[self.streets.index(street)].acquire()
                self.streets_lock[self.streets.index(new_street)].acquire()

            sleep(delay)

            self.streets_lock[self.streets.index(street)].release()
            self.streets_lock[self.streets.index(new_street)].release()

    def __str__(self) -> str:
        text = f'Crossing: index {self.index}. n_streets {self.n_streets}. streets {self.streets}.'

        return text


class Street:
    index: int
    size: int
    crossings: Tuple[int, int]
    capacities: Tuple[int, int]

    populations: List[int]

    #butler

    def __init__(
        self,
        index: int,
        size: int,
        crossings: Tuple[int, int],
        capacities: Tuple[int, int],
    ):

        self.index = index
        self.size = size
        self.crossings = crossings
        self.capacities = capacities

        self.populations = [0, 0]
        self.butler = Lock()

    def __str__(self) -> str:
        text = f'Street: index {self.index}. size {self.size}. crossings {self.crossings}. capacities {self.capacities}. populations {self.populations}'

        return text

    def enter(self, direction: int) -> bool:
        with self.butler:
            if self.populations[direction] < self.capacities[direction]:
                self.populations[direction] += 1

                return True

        return False

    def exit(self, direction: int) -> bool:
        with self.butler:
            self.populations[direction] -= 1

        return True


class Car:
    index: int
    on_street: int
    direction: int
    get_velocity: Callable[[], int]

    position: int
    velocity: int
    odometer: int
    destiny: Optional[int]

    def __init__(
        self,
        index: int,
        on_street: int,
        direction: int,
        get_velocity: Callable[[], int],
    ):

        self.index = index
        self.on_street = on_street
        self.direction = direction
        self.get_velocity = get_velocity

        self.position = 0
        self.velocity = self.get_velocity()
        self.odometer = 0
        self.destiny = None

    def update(self, crossings: List[Crossing], streets: List[Street]) -> Optional[car_result]:
        street = [street for street in streets if street.index == self.on_street][0]

        if self.position == street.size:
            on_crossing = street.crossings[self.direction]
            crossing = crossings[on_crossing]

            new_on_street = choice(crossing.streets) if self.destiny is None else self.destiny
            new_street = streets[new_on_street]
            new_direction = 1 if new_street.crossings[0] == on_crossing else 0

            if new_street.enter(new_direction) and street.exit(self.direction):
                crossing.to_cross(self.on_street, new_on_street)

                if new_on_street < 0:
                    return None

                self.on_street = new_on_street
                self.direction = new_direction
                self.position = 0
                self.velocty = self.get_velocity()
                self.destiny = None

            else:
                self.destiny = new_on_street

        else:
            distance = min(self.velocity, street.size - self.position)
            self.position += distance
            self.odometer += distance

        result = self.index, self.on_street, self.velocity, self.odometer

        return result

    def __str__(self):
        text = f'Car: index {self.index}. on_street {self.on_street}. direction {self.direction}. position {self.position}. velocity {self.velocity}. odometer {self.odometer}.'

        return text


class City:
    n_processes: int
    graph: nx.classes.graph.Graph
    n_terminal_streets: int
    get_new_cars: Callable[[], int]
    get_velocity: Callable[[], int]

    index: str
    steps: int
    n_crossings: int
    crossings: List[Crossing]
    n_streets: int
    streets: List[Street]
    n_cars: int
    cars: List[Car]
    index_new_car: int
    # butler
    car_index: int

    def __init__(
        self,
        n_processes: int,
        n_crossing: int,
        prob_edge_creat: float,
        n_terminal_streets: int,
        n_init_cars: int,
        get_new_cars: Callable[[], int],
        get_size: Callable[[], int],
        get_capacities: Callable[[], Tuple[int, int]],
        get_velocity: Callable[[], int],
    ) -> None:

        self.n_processes = n_processes
        self.graph = erdos_renyi_graph(n_crossing, prob_edge_creat)
        self.n_terminal_streets = n_terminal_streets
        self.get_new_cars = get_new_cars
        self.get_velocity = get_velocity

        self.index = str(uuid.uuid1()).replace('-', '_')
        self.steps = 0
        self.n_crossings = 0
        self.crossings = []
        self.n_streets = 0
        self.streets = []
        self.n_cars = 0
        self.cars = []
        self.index_new_car = 0
        self.butler = Lock()
        self.car_index = 0

        for index in self.graph.nodes():
            if not isinstance(index, int):
                continue

            crossing = Crossing(index)
            self.n_crossings += 1
            self.crossings.append(crossing)

            nx.set_node_attributes(self.graph, {index: index}, name='index')

        for index, (left_crossing, right_crossing, _) in enumerate(self.graph.edges(data=True)):
            street = Street(index, get_size(), (left_crossing, right_crossing), get_capacities())
            self.n_streets += 1
            self.streets.append(street)

            self.crossings[left_crossing].add_street(index)
            self.crossings[right_crossing].add_street(index)

            nx.set_edge_attributes(self.graph, {(left_crossing, right_crossing): index},
                                   name='index')

        for index in range(-1, -self.n_terminal_streets - 1, -1):
            crossing = randint(0, self.n_crossings - 1)
            street = Street(index, 0, (index, crossing), (np.inf, np.inf))
            self.n_streets += 1
            self.streets.append(street)
            self.crossings[crossing].add_street(index)
            self.graph.add_edge(index, crossing, index=index)

        for _ in range(n_init_cars):
            self.add_car()

    def add_car(self) -> None:
        street = randint(-self.n_terminal_streets, -1)
        car = Car(self.index_new_car, street, 1, self.get_velocity)
        self.n_cars += 1
        self.cars.append(car)
        self.index_new_car += 1
        self.streets[street].enter(1)

    def update(self, _) -> List[car_result]:
        car_results = []

        while True:
            with self.butler:
                car_index = self.car_index
                self.car_index += 1

            if car_index >= self.n_cars:
                return car_results

            result = self.cars[car_index].update(self.crossings, self.streets)

            if result is not None:
                car_results.append(result)

    def run(self) -> city_result:
        self.steps += 1

        for _ in range(self.get_new_cars()):
            self.add_car()

        self.car_index = 0

        car_results = []

        with ThreadPool(processes=self.n_processes) as pool:
            results = pool.map(self.update, range(self.n_processes))
            for result in results:
                car_results += result

        car_results_index = set(result[0] for result in car_results)

        self.cars = [car for car in self.cars if car.index in car_results_index]
        self.n_cars = len(self.cars)

        result = (self.index, self.steps, self.n_crossings, self.n_streets, self.n_cars,
                  car_results)

        return result

    def __str__(self) -> str:
        text = ('\n'.join(str(crossing) for crossing in self.crossings) + '\n\n' +
                '\n'.join(str(street) for street in self.streets) + '\n\n' +
                '\n'.join(str(car) for car in self.cars))

        return text

    def draw(self) -> None:
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos)
        node_labels = nx.get_node_attributes(self.graph, 'index')
        nx.draw_networkx_labels(self.graph, pos, node_labels)
        edge_labels = nx.get_edge_attributes(self.graph, 'index')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels)
        plt.show()


# -----
