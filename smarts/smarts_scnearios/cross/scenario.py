# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from pathlib import Path

import smarts.sstudio.types as t
from smarts.sstudio import gen_scenario
import os
import random
from pathlib import Path
import numpy as np

from smarts.sstudio import gen_traffic, gen_missions, gen_social_agent_missions, gen_scenario
from smarts.sstudio.types import (
    Scenario,
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    SocialAgentActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
    EndlessMission,
)

scenario = os.path.dirname(os.path.realpath(__file__))

missions = [
    t.Mission(t.Route(begin=("gneE17", 0, 10),end=('gneE23',0,'max')),start_time=15)
    # t.Mission(t.Route(begin=("gneE22", 0, 10), end=("gneE5", 1, 100)),start_time=30),
    # t.Mission(t.Route(begin=("gneE17", 0, 25), end=("gneE5", 0, 100)),start_time=30),
    # t.Mission(t.Route(begin=("gneE22", 0, 25), end=("gneE5", 1, 100)),start_time=30),
]

# impatient_car = t.TrafficActor(
#     name="car",
#     speed=t.Distribution(sigma=0.2, mean=1.0),
#     lane_changing_model=t.LaneChangingModel(impatience=1, cooperative=0.25),
# )

# patient_car = t.TrafficActor(
#     name="car",
#     speed=t.Distribution(sigma=0.2, mean=0.8),
#     lane_changing_model=t.LaneChangingModel(impatience=0, cooperative=0.5),
# )
# start_routes = ["gneE17", "gneE22"]
# end_routes = ["gneE5"]


# traffic = {
#     "1": t.Traffic(
#         flows=[
#             t.Flow(
#                 route=t.Route(
#                     begin=(f"gneE22", 0, 60),
#                     end=(f"gneE22", 0, 100),
#                 ),
#                 rate=1,
#                 actors={impatient_car: 0.5, patient_car: 0.5},
#             )
#         ]
#     ),
# }
for seed in np.random.choice(1000, 40, replace=False):
    actors = {}

    for i in range(4):
        car = TrafficActor(
            name = f'car_type_{i+1}',
            speed=Distribution(mean=np.random.uniform(0.6, 1.0), sigma=0.1),
            min_gap=Distribution(mean=np.random.uniform(2, 4), sigma=0.1),
            imperfection=Distribution(mean=np.random.uniform(0.3, 0.7), sigma=0.1),
            lane_changing_model=LaneChangingModel(speed_gain=np.random.uniform(1.0, 2.0), impatience=np.random.uniform(0, 1.0), cooperative=np.random.uniform(0, 1.0)),
            junction_model=JunctionModel(ignore_foe_prob=np.random.uniform(0, 1.0), impatience=np.random.uniform(0, 1.0)),
        )

        actors[car] = 1/4

    flows = []

    # flows.append(Flow(route=Route(begin=('gneE17', 0,90), end=('gneE23', 0,'max')), rate=100, actors=actors))
    flows.append(Flow(route=Route(begin=('gneE17', 0,80), end=('gneE24', 0,'max')), rate=80, actors=actors))
    flows.append(Flow(route=Route(begin=('gneE22', 0,"random"), end=('gneE24', 0,'max')), rate=200, actors=actors))
    flows.append(Flow(route=Route(begin=('gneE22', 0,"random"), end=('gneE23', 0,'max')), rate=200, actors=actors))

    traffic = Traffic(flows=flows)
    gen_traffic(scenario, traffic, seed=seed, name=f'traffic_{seed}')

gen_missions(
    scenario=scenario,
    missions=missions
)
