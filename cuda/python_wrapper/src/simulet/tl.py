from enum import Enum
from typing import List

from .convert import PbTl, save_pb


class LightState(Enum):
    UNSPECIFIED = PbTl.LIGHT_STATE_UNSPECIFIED
    RED = PbTl.LIGHT_STATE_RED
    GREEN = PbTl.LIGHT_STATE_GREEN
    YELLOW = PbTl.LIGHT_STATE_YELLOW


class LightPhase:
    def __init__(self, pb):
        self.pb = pb
        self.duration: float = pb.duration
        self.states: List[LightState] = [LightState(i) for i in pb.states]


class TrafficLight:
    def __init__(self, pb):
        self.pb = pb
        self.junction_id: int = pb.junction_id
        self.phases: List[LightPhase] = [LightPhase(i) for i in pb.phases]


class TrafficLights:
    def __init__(self, tl_file):
        self.pb = tl = PbTl.TrafficLights()
        tl.ParseFromString(open(tl_file, 'rb').read())
        self.tls = [TrafficLight(i) for i in tl.traffic_lights]
        self.tls_map = {i.junction_id: i for i in self.tls}

    def save(self, tl_file):
        del self.pb.traffic_lights[:]
        for i in self.tls:
            self.pb.traffic_lights.append(i.pb)
        save_pb(self.pb, tl_file)
