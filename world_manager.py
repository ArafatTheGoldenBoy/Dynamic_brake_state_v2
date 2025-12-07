import math
import random
import time
from typing import Optional

from carla_utils import import_carla
from config import DT, LATERAL_MAX

carla = import_carla()


class WorldManager:
    def __init__(self, host: str, port: int, town: Optional[str], mu: float, apply_tire_friction: bool,
                 npc_vehicles: int = 0, npc_walkers: int = 0, npc_seed: Optional[int] = None,
                 npc_autopilot: bool = True, npc_speed_diff_pct: int = 10):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        if town:
            load_town = town
            try:
                avail = []
                try:
                    avail = list(self.client.get_available_maps())
                except Exception:
                    avail = []

                def has_map(name: str) -> bool:
                    try:
                        return any(m.endswith('/' + name) or m.endswith('\\' + name) or m == name for m in avail)
                    except Exception:
                        return False

                if not town.endswith('_Opt'):
                    opt = town + '_Opt'
                    if has_map(opt):
                        print(f"[INFO] Auto-switching to optimized map '{opt}' (was '{town}').")
                        load_town = opt
                else:
                    base = town[:-4]
                    if not has_map(town) and has_map(base):
                        print(f"[INFO] Optimized map '{town}' not found; falling back to '{base}'.")
                        load_town = base
            except Exception:
                load_town = town
            try:
                w0 = self.client.get_world()
                s0 = w0.get_settings()
                if s0.synchronous_mode:
                    s0.synchronous_mode = False
                    w0.apply_settings(s0)
            except Exception:
                pass
            try:
                self.world = self.client.load_world(load_town, map_layers=carla.MapLayer.All)
            except TypeError:
                self.world = self.client.load_world(load_town)
            try:
                if hasattr(carla, 'MapLayer'):
                    to_remove = (carla.MapLayer.ParkedVehicles | carla.MapLayer.Props)
                    self.world.unload_map_layer(to_remove)
                    try:
                        self.world.tick()
                    except Exception:
                        pass
                    try:
                        map_name = self.world.get_map().name
                        if not map_name.endswith('_Opt'):
                            print(f"[INFO] Map '{map_name}' is not an _Opt variant; baked parked vehicles may remain. Try --town {map_name}_Opt if available.")
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            self.world = self.client.get_world()
            try:
                if hasattr(carla, 'MapLayer'):
                    to_remove = (carla.MapLayer.ParkedVehicles | carla.MapLayer.Props)
                    self.world.unload_map_layer(to_remove)
                    try:
                        self.world.tick()
                    except Exception:
                        pass
                    try:
                        map_name = self.world.get_map().name
                        if not map_name.endswith('_Opt'):
                            print(f"[INFO] Map '{map_name}' is not an _Opt variant; baked parked vehicles may remain. Consider switching to an _Opt map.")
                    except Exception:
                        pass
            except Exception:
                pass

        self.original_settings = self.world.get_settings()
        self.tm = self.client.get_trafficmanager()
        self.map = self.world.get_map()
        print(f"Loaded map: {self.map.name}")

        try:
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        except Exception:
            pass

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)

        self.npc_vehicles = []
        self.npc_walkers = []
        self.walker_controllers = []
        self.collision_sensor = None
        self.collision_happened = False
        self.collision_last_actor = None
        self.collision_last_impulse = None
        self.collision_last_time = None

        bp_lib = self.world.get_blueprint_library()
        try:
            ego_bp = bp_lib.filter('vehicle.taxi.ford')[0]
        except Exception:
            ego_bp = bp_lib.find('vehicle.tesla.model3')
        spawn = carla.Transform(
            carla.Location(x=95.70, y=66.00, z=2.00),
            carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0),
        )
        ego = self.world.try_spawn_actor(ego_bp, spawn)
        if ego is None:
            ego = self.world.spawn_actor(ego_bp, random.choice(self.map.get_spawn_points()))
        ego.set_autopilot(False)
        self.ego = ego

        try:
            col_bp = bp_lib.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                col_bp,
                carla.Transform(),
                attach_to=self.ego,
            )

            def _on_collision(event):  # type: ignore[arg-type]
                other_name = 'unknown'
                impulse_mag = None
                try:
                    other = getattr(event, 'other_actor', None)
                    if other is not None:
                        other_name = getattr(other, 'type_id', str(other))
                except Exception:
                    pass
                try:
                    impulse = getattr(event, 'normal_impulse', None)
                    if impulse is not None:
                        impulse_mag = math.sqrt(
                            float(impulse.x) ** 2 + float(impulse.y) ** 2 + float(impulse.z) ** 2
                        )
                except Exception:
                    impulse_mag = None
                self.collision_happened = True
                self.collision_last_actor = other_name
                self.collision_last_impulse = impulse_mag
                self.collision_last_time = time.time()

                try:
                    if impulse_mag is not None:
                        print(f"[COLLISION] Contact with {other_name} | impulse={impulse_mag:.1f}")
                    else:
                        print(f"[COLLISION] Contact with {other_name}")
                except Exception:
                    pass

            self.collision_sensor.listen(_on_collision)
        except Exception:
            self.collision_sensor = None

        try:
            phys = self.ego.get_physics_control()
            for w in phys.wheels:
                w.max_brake_torque = max(8000.0, getattr(w, 'max_brake_torque', 4000.0))
                w.max_handbrake_torque = max(12000.0, getattr(w, 'max_handbrake_torque', 8000.0))
                if apply_tire_friction:
                    w.tire_friction = mu
            self.ego.apply_physics_control(phys)
        except Exception:
            pass

        try:
            if npc_seed is not None:
                random.seed(int(npc_seed))
            if npc_vehicles > 0:
                self._spawn_npc_vehicles(
                    npc_vehicles,
                    npc_autopilot,
                    int(max(0, min(100, npc_speed_diff_pct))),
                )
            if npc_walkers > 0:
                self._spawn_npc_walkers(npc_walkers)
        except Exception:
            pass

    def tick(self, block: bool = True):
        if block:
            return self.world.tick()
        else:
            snap = self.world.wait_for_tick()
            return getattr(snap, 'frame', snap)

    def destroy(self):
        try:
            if self.world is not None:
                self.world.apply_settings(self.original_settings)
        except Exception:
            pass
        try:
            self.tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
        except Exception:
            pass
        try:
            if self.ego is not None:
                self.ego.destroy()
        except Exception:
            pass
        try:
            if self.npc_vehicles:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_vehicles])
        except Exception:
            pass
        try:
            if self.walker_controllers:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_controllers])
            if self.npc_walkers:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_walkers])
        except Exception:
            pass

    def lead_distance_ahead(
        self,
        lateral_max: float = LATERAL_MAX + 0.5,
        max_distance: float = 150.0,
    ) -> Optional[float]:
        """Return distance to the closest vehicle or walker ahead of the ego.

        The search is done in ego-vehicle local coordinates:

        - Only actors with longitudinal position > 0 (in front of ego) are considered.
        - Lateral offset must be within ±``lateral_max`` meters.
        - Only CARLA ``vehicle.*`` and ``walker.*`` actors are included.

        Returns
        -------
        float | None
            Euclidean distance in meters to the closest qualifying actor,
            or None if no suitable lead is found or on error.
        """
        try:
            ego = self.ego
            if ego is None:
                return None
            ego_loc = ego.get_location()
            ego_rot = ego.get_transform().rotation
        except Exception:
            return None

        try:
            actors = self.world.get_actors()
        except Exception:
            return None

        yaw = math.radians(ego_rot.yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        best_dist: Optional[float] = None
        for actor in actors:
            try:
                if actor.id == ego.id:
                    continue
                tid = getattr(actor, "type_id", "").lower()
                if not (tid.startswith("vehicle.") or tid.startswith("walker.")):
                    continue
                loc = actor.get_location()
            except Exception:
                continue

            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            longitudinal = dx * cos_yaw + dy * sin_yaw
            if longitudinal <= 0.0:
                # Actor is behind or exactly alongside; not a lead
                continue

            lateral = -dx * sin_yaw + dy * cos_yaw
            if abs(lateral) > lateral_max:
                continue

            dist = math.hypot(dx, dy)
            if dist > max_distance:
                continue

            if best_dist is None or dist < best_dist:
                best_dist = dist

        return best_dist

    def _spawn_npc_vehicles(self, count: int, autopilot: bool, speed_diff_pct: int = 10):
        bp_lib = self.world.get_blueprint_library()
        vehicles = []
        for v in bp_lib.filter('vehicle.*'):
            if not v.id.startswith('vehicle.motorbike') and not v.id.startswith('vehicle.bh.crossbike'):
                vehicles.append(v)
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        spawned = 0
        for tr in spawn_points:
            if spawned >= count:
                break
            bp = random.choice(vehicles)
            try:
                v = self.world.try_spawn_actor(bp, tr)
                if v is None:
                    continue
                v.set_autopilot(autopilot)
                try:
                    try:
                        v.set_light_state(carla.VehicleLightState.LowBeam)
                    except Exception:
                        pass
                    if hasattr(self.tm, 'distance_to_leading_vehicle'):
                        self.tm.distance_to_leading_vehicle(v, 1.0)
                    if hasattr(self.tm, 'global_percentage_speed_difference'):
                        self.tm.global_percentage_speed_difference(v, speed_diff_pct)
                except Exception:
                    pass
                spawned += 1
                self.npc_vehicles.append(v)
            except Exception:
                pass

    def _spawn_npc_walkers(self, count: int):
        bp_lib = self.world.get_blueprint_library()
        walkers = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        spawn_points = []
        for _ in range(count):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(carla.Transform(loc))
        batch = [carla.command.SpawnActor(random.choice(walkers), sp) for sp in spawn_points]
        results = self.client.apply_batch_sync(batch, True)
        spawned = 0
        for res, sp in zip(results, spawn_points):
            if res.error:
                continue
            actor_id = res.actor_id
            try:
                w = self.world.try_spawn_actor(bp_lib.find(res.type_id), sp)
                if w is None:
                    continue
                ctrl = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
                ctrl.start()
                try:
                    ctrl.go_to_location(self.world.get_random_location_from_navigation())
                    ctrl.set_max_speed(1.4 + random.uniform(-0.3, 0.6))
                except Exception:
                    pass
                self.npc_walkers.append(w)
                self.walker_controllers.append(ctrl)
                spawned += 1
            except Exception:
                pass
