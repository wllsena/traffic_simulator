# -*- coding: utf-8 -*-
__author__ = "William Sena <@wllsena>"
"""
Style Guide: PEP 8. Column limit: 100.
Author: William Sena <@wllsena>.
"""

from random import choice, randint
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph

# -----

# index, on_street, velocity, odometer
car_result = Tuple[int, int, int, int]

# (self.n_crossing, self.n_streets, self.n_cars, car_results)
city_result = Tuple[int, int, int, List[car_result]]


class Crossing:
    index: int
    n_streets: int
    streets: List[int]

    def __init__(self, index: int):
        self.index = index

        self.n_streets = 0
        self.streets = []

    def add_street(self, street: int) -> None:
        self.n_streets += 1
        self.streets.append(street)

    def to_cross(self, index, street, new_street) -> None:
        return None

    def __str__(self) -> str:
        text = f'Crossing: index {self.index}. n_streets {self.n_streets}. streets {self.streets}.'

        return text


class Street:
    index: int
    size: int
    crossings: Tuple[int, int]
    capacities: Tuple[int, int]
    populations: List[int]

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

    def __str__(self) -> str:
        text = f'Street: index {self.index}. size {self.size}. crossings {self.crossings}. capacities {self.capacities}. populations {self.populations}'

        return text

    def add_car(self, direction: int) -> None:
        self.populations[direction] += 1


class Car:
    index: int
    on_street: int
    direction: int
    position: int
    get_velocity: Callable[[], int]
    velocity: int
    odometer: int
    destiny: Optional[int]

    def __init__(
        self,
        index: int,
        on_street: int,
        direction: int,
        position: int,
        get_velocity: Callable[[], int],
    ):

        self.index = index
        self.on_street = on_street
        self.direction = direction
        self.position = position

        self.get_velocity = get_velocity
        self.velocity = self.get_velocity()
        self.odometer = 0
        self.destiny = None

    def update(self, crossings: List[Crossing], streets: List[Street]) -> Optional[car_result]:
        street = streets[self.on_street]

        if self.position == street.size:
            on_crossing = street.crossings[self.direction]
            crossing = crossings[on_crossing]

            new_on_street = choice(crossing.streets) if self.destiny is None else self.destiny
            new_street = streets[new_on_street]
            new_direction = 1 if new_street.crossings[0] == on_crossing else 0

            if new_street.populations[new_direction] < new_street.capacities[new_direction]:
                crossing.to_cross(self.index, self.on_street, new_on_street)

                if new_on_street < 0:
                    return None

                street.populations[self.direction] -= 1
                new_street.populations[new_direction] += 1

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
    graph: nx.classes.graph.Graph
    n_terminal_streets: int
    n_streets: int
    crossings: List[Crossing]
    n_streets: int
    streets: List[Street]
    n_cars: int
    cars: List[Car]
    index_car: int

    def __init__(
        self,
        n_crossing: int,
        prob_edge_creat: float,
        n_terminal_streets: int,
        get_size: Callable[[], int],
        get_capacities: Callable[[], Tuple[int, int]],
    ) -> None:

        self.graph = erdos_renyi_graph(n_crossing, prob_edge_creat)
        self.n_terminal_streets = n_terminal_streets

        self.n_crossings = 0
        self.crossings = []
        self.n_streets = 0
        self.streets = []
        self.n_cars = 0
        self.cars = []
        self.index_car = 0

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

    def add_car(self, get_velocity: Callable[[], int]) -> None:
        street = randint(-self.n_terminal_streets, -1)
        car = Car(self.index_car, street, 1, 0, get_velocity)
        self.n_cars += 1
        self.cars.append(car)
        self.index_car += 1
        self.streets[street].add_car(1)

    def update(self) -> city_result:
        car_results = []

        new_cars = []

        for car in self.cars:
            car_result = car.update(self.crossings, self.streets)

            if car_result is not None:
                car_results.append(car_result)
                new_cars.append(car)

        self.n_cars = len(new_cars)
        self.cars = new_cars

        result = (self.n_crossings, self.n_streets, self.n_cars, car_results)

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
