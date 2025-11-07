#ifndef SRC_PROTO_H_
#define SRC_PROTO_H_

#include "wolong/agent/v2/agent.pb.h"
#include "wolong/geo/v2/geo.pb.h"
#include "wolong/map/v2/map.pb.h"
#include "wolong/routing/v2/routing.pb.h"
#include "wolong/routing/v2/routing_service.grpc.pb.h"
#include "wolong/routing/v2/routing_service.pb.h"
#include "wolong/traffic_light/v2/traffic_light.pb.h"
#include "wolong/trip/v2/trip.pb.h"

namespace simulet {
// 地图
using PbAoi = wolong::map::v2::Aoi;
using PbMap = wolong::map::v2::Map;
using PbTls = wolong::traffic_light::v2::TrafficLights;
using PbTl = wolong::traffic_light::v2::TrafficLight;
using PbLane = wolong::map::v2::Lane;
using PbLaneType = wolong::map::v2::LaneType;
using PbRoad = wolong::map::v2::Road;
using PbJunction = wolong::map::v2::Junction;
using PbAgents = wolong::agent::v2::Agents;
using PbAgent = wolong::agent::v2::Agent;
using PbSchedule = wolong::trip::v2::Schedule;
using AgentType = wolong::agent::v2::AgentType;
using TripMode = wolong::trip::v2::TripMode;
using LightState = wolong::traffic_light::v2::LightState;

// 导航
using PbGetRouteRequest = wolong::routing::v2::GetRouteRequest;
using PbGetRouteResponse = wolong::routing::v2::GetRouteResponse;
using LaneType = wolong::map::v2::LaneType;
using LaneTurn = wolong::map::v2::LaneTurn;
using RouteType = wolong::routing::v2::RouteType;
using MovingDirection = wolong::routing::v2::MovingDirection;
using JourneyType = ::wolong::routing::v2::JourneyType;
// using AgentPbGetRouteRequest =wolong::routing::v2::getr

// 交通灯
using LightState = wolong::traffic_light::v2::LightState;
using PbLight = wolong::traffic_light::v2::TrafficLight;
using PbLights = wolong::traffic_light::v2::TrafficLights;
using PbPhase = wolong::traffic_light::v2::Phase;

using PbPosition = ::wolong::geo::v2::Position;
}  // namespace simulet
#endif
